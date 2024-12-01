import os
import tqdm
import time
import math
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTPretrainTrainer(nn.Module):
    """
    BERT Pretrain Model trainer

    :param model: BERT Pretrain model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, model, optimizer, config_dict):
        super(BERTPretrainTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict

    def train_one_epoch(self, data_loader, epoch):
        """
        Train step

        :param data_loader: Train Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :param epoch: Epoch number
        :type epoch: int
        :return: Train Losses (Train Loss, Train masked tokens loss, Train NSP loss)
        :rtype: tuple (torch.float32, torch.flooat32, torch.float32)
        """
        self.model.train()
        total_loss, total_clf_loss, total_nsp_loss, num_instances = 0, 0, 0, 0

        self.logger.info(
            f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------"
        )
        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
        )

        for batch_id, (tokens, tokens_mask, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)

            clf_loss, nsp_loss = self.calc_loss(
                tokens_pred, nsp_pred, tokens, nsp_labels, tokens_mask
            )
            loss = clf_loss + nsp_loss
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            total_clf_loss += clf_loss
            total_nsp_loss += nsp_loss
            num_instances += tokens.size(0)

        train_loss = total_loss / num_instances
        train_clf_loss = total_clf_loss / num_instances
        train_nsp_loss = total_nsp_loss / num_instances

        return train_loss, train_clf_loss, train_nsp_loss

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        """
        Validation step

        :param data_loader: Validation Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :return: Validation Losses (Validation Loss, Validation masked tokens loss, Validation NSP loss)
        :rtype: tuple (torch.float32, torch.flooat32, torch.float32)
        """
        self.model.eval()
        total_loss, total_clf_loss, total_nsp_loss, num_instances = 0, 0, 0, 0

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Validation"
        )

        for batch_id, (tokens, tokens_mask, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)

            clf_loss, nsp_loss = self.calc_loss(
                tokens_pred, nsp_pred, tokens, nsp_labels, tokens_mask
            )
            loss = clf_loss + nsp_loss

            total_loss += loss
            total_clf_loss += clf_loss
            total_nsp_loss += nsp_loss
            num_instances += tokens.size(0)

        val_loss = total_loss / num_instances
        val_clf_loss = total_clf_loss / num_instances
        val_nsp_loss = total_nsp_loss / num_instances

        return val_loss, val_clf_loss, val_nsp_loss

    @torch.no_grad()
    def predict(self, data_loader):
        """
        Runs inference on input data

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: Labels, Predictions (True tokens, NSP Labels, Predicton tokens, NSP Predictions)
        :rtype: tuple (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples,], numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples,])
        """
        self.model.eval()
        y_tokens_pred, y_tokens_true = [], []
        y_nsp_pred, y_nsp_true = [], []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, (tokens, _, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)

            y_tokens_pred.append(tokens_pred.cpu().detach().numpy())
            y_tokens_true.append(tokens.cpu().detach().numpy())

            y_nsp_true.append(nsp_labels.cpu().detach().numpy())
            y_nsp_pred.append(nsp_pred.cpu().detach().numpy())

        y_tokens_pred = np.concatenate(y_tokens_pred, axis=0)
        y_tokens_true = np.concatenate(y_tokens_true, axis=0)
        y_nsp_pred = np.concatenate(y_nsp_pred, axis=0)
        y_nsp_true = np.concatenate(y_nsp_true, axis=0)

        return y_tokens_true, y_nsp_true, y_tokens_pred, y_nsp_pred

    def fit(self, train_loader, val_loader):
        """
        Fits the model on dataset. Runs training and Validation steps for given epochs and saves best model based on the evaluation metric

        :param train_loader: Train Data loader
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: Validaion Data Loader
        :type val_loader: torch.utils.data.DataLoader
        :return: Training History
        :rtype: dict
        """
        num_epochs = self.config_dict["train"]["epochs"]
        output_folder = self.config_dict["paths"]["output_folder"]

        best_val_loss = np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_clf_loss, train_nsp_loss = self.train_one_epoch(
                train_loader, epoch
            )
            val_loss, val_clf_loss, val_nsp_loss = self.val_one_epoch(val_loader)

            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            history["train_clf_loss"].append(float(train_clf_loss.detach().numpy()))
            history["val_clf_loss"].append(float(val_clf_loss.detach().numpy()))
            history["train_nsp_loss"].append(float(train_nsp_loss.detach().numpy()))
            history["val_nsp_loss"].append(float(val_nsp_loss.detach().numpy()))

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")

            if val_loss <= best_val_loss:
                self.logger.info(
                    f"Validation Loss score improved from {best_val_loss} to {val_loss}"
                )
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_folder, "best_model_pretrain.pt"),
                )
            else:
                self.logger.info(
                    f"Validation Loss score didn't improve from {best_val_loss}"
                )

        end = time.time()
        time_taken = end - start
        self.logger.info(
            "Training completed in {:.0f}h {:.0f}m {:.0f}s".format(
                time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60
            )
        )
        self.logger.info(f"Best Val Loss score: {best_val_loss}")

        return history

    def calc_loss(self, tokens_pred, nsp_pred, tokens_true, nsp_labels, tokens_mask):
        """
        Calculates Training Loss components

        :param tokens_pred: Predicted Tokens
        :type tokens_pred: torch.Tensor (batch_size, seq_len, num_vocab)
        :param nsp_pred: Predicted NSP Label
        :type nsp_pred: torch.Tensor (batch_size,)
        :param tokens_true: True tokens
        :type tokens_true: torch.Tensor (batch_size, seq_len)
        :param nsp_labels: NSP labels
        :type nsp_labels: torch.Tensor (batch_size,)
        :param tokens_mask: Tokens mask
        :type tokens_mask: torch.Tensor (batch_size, seq_len)
        :return: Masked word prediction Cross Entropy loss, NSP classification loss
        :rtype: tuple (torch.float32, torch.float32)
        """
        num_vocab = self.config_dict["dataset"]["num_vocab"]
        nsp_loss_fn = nn.BCELoss()
        nsp_loss = nsp_loss_fn(nsp_pred.squeeze(), nsp_labels)

        mask_flatten = torch.flatten(tokens_mask).bool()
        mask_flatten = torch.stack([mask_flatten] * num_vocab, dim=1)

        y_pred = torch.flatten(tokens_pred, end_dim=1)
        y_true = torch.flatten(tokens_true)
        y_true = F.one_hot(y_true, num_classes=num_vocab).to(torch.float)

        y_pred_mask = torch.masked_select(y_pred, mask_flatten).reshape(
            -1, y_pred.shape[1]
        )
        y_true_mask = torch.masked_select(y_true, mask_flatten).reshape(
            -1, y_true.shape[1]
        )

        clf_loss_fn = nn.CrossEntropyLoss(reduce="sum")
        clf_loss = clf_loss_fn(y_pred_mask, y_true_mask)

        return clf_loss, nsp_loss
