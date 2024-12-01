import os
import tqdm
import time
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import ClassificationMetrics


class GRUCell(nn.Module):
    """
    GRU Cell

    :param h_dim: Hidden state vector dimension
    :type h_dim: int
    :param inp_x_dim: Input vector dimension
    :type inp_x_dim: int
    :param out_x_dim: Output vector dimension
    :type out_x_dim: int
    """

    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(GRUCell, self).__init__()

        self.zh_dense = nn.Linear(h_dim, h_dim)
        self.zx_dense = nn.Linear(inp_x_dim, h_dim)

        self.rh_dense = nn.Linear(h_dim, h_dim)
        self.rx_dense = nn.Linear(inp_x_dim, h_dim)

        self.uh_dense = nn.Linear(h_dim, h_dim)
        self.ux_dense = nn.Linear(inp_x_dim, h_dim)

        self.xh_dense = nn.Linear(h_dim, out_x_dim)

    def forward(self, ht_1, xt):
        """
        Forward propogation

        :param ht_1: Hidden state vector
        :type ht_1: torch.Tensor (batch_size, h_dim)
        :param xt: Input vector
        :type xt: torch.Tensor (batch_size, embed_dim)
        :return: New hidden states, output
        :rtype: tuple (torch.Tensor [batch_size, h_dim], torch.Tensor [batch_size, out_dim])
        """
        zt = nn.Sigmoid()(self.zh_dense(ht_1) + self.zx_dense(xt))
        rt = nn.Sigmoid()(self.rh_dense(ht_1) + self.rx_dense(xt))

        ht_ = nn.Tanh()(self.uh_dense(rt * ht_1) + self.ux_dense(xt))
        ht = (1 - zt) * ht_1 + zt * ht_

        yt = self.xh_dense(ht)

        return ht, yt


class GRUModel(nn.Module):
    """
    GRU Architecture

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(GRUModel, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.h_dim = config_dict["model"]["h_dim"][0]
        self.x_dim = config_dict["model"]["x_dim"][0]

        num_vocab = config_dict["dataset"]["num_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        num_classes = config_dict["dataset"]["num_classes"]
        self.embed_layer = nn.Embedding(num_vocab, embed_dim)

        self.gru_cell = GRUCell(self.h_dim, embed_dim, self.x_dim)

        self.classifier_layer = nn.Linear(self.x_dim, num_classes)

    def forward(self, X):
        """
        Forward Propogation

        :param X: Input tokens
        :type X: torch.Tensor (batch_size, seq_len)
        :return: Predicted labels
        :rtype: toch.Tensor (batch_size, seq_len, num_classes)
        """
        x_embed = self.embed_layer(X.to(torch.long))
        self.num_samples = X.size(0)

        ht = self.init_hidden()
        yprobs = []

        for i in range(self.seq_len):
            ht, yt = self.gru_cell(ht, x_embed[:, i, :])
            # yprob = nn.Softmax()(self.classifier_layer(yt))
            yprob = self.classifier_layer(yt)
            yprobs.append(yprob[:, None, :])

        return torch.concat(yprobs, dim=1)

    def init_hidden(self):
        """
        Initialized hidden state

        :return: Hidden state
        :rtype: torch.Tensor (num_samples, h_dim)
        """
        ht = nn.init.kaiming_uniform_(torch.empty(self.num_samples, self.h_dim))

        return ht


class GRUTrainer(nn.Module):
    """
    GRU Trainer

    :param model: GRU model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, model, optimizer, config_dict):
        super(GRUTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict
        self.metric_cls = ClassificationMetrics(config_dict)
        self.eval_metric = config_dict["train"]["eval_metric"]
        self.target_names = list(config_dict["dataset"]["labels"])

    def train_one_epoch(self, data_loader, epoch):
        """
        Train step

        :param data_loader: Train Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :param epoch: Epoch number
        :type epoch: int
        :return: Train Losse, Train Metrics
        :rtype: tuple (torch.float32, dict)
        """
        self.model.train()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        self.logger.info(
            f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------"
        )
        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
        )

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.calc_loss(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(-1).flatten())
            y_pred.append(y_hat.cpu().detach().numpy().reshape(-1, y_hat.shape[-1]))

        train_loss = total_loss / num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        train_metrics = self.metric_cls.get_metrics(y_true, y_pred, self.target_names)

        return train_loss, train_metrics

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        """
        Validation step

        :param data_loader: Validation Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :return: Validation Losse, Validation Metrics
        :rtype: tuple (torch.float32, dict)
        """
        self.model.eval()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Validation"
        )

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.calc_loss(y_hat, y)

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(-1).flatten())
            y_pred.append(y_hat.cpu().detach().numpy().reshape(-1, y_hat.shape[-1]))

        val_loss = total_loss / num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_metrics = self.metric_cls.get_metrics(y_true, y_pred, self.target_names)

        return val_loss, val_metrics

    @torch.no_grad()
    def predict(self, data_loader):
        """
        Runs inference to predict a translation of soruce sentence

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: Predicted labels
        :rtype: numpy.ndarray (num_samples, seq_len)
        """
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, X in pbar:
            tokens_hat = self.model(X[0])
            y_pred.append(tokens_hat.cpu().detach().numpy().argmax(-1).flatten())

        y_pred = np.concatenate(y_pred, axis=0)

        return y_pred

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

        best_val_metric = -np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.val_one_epoch(val_loader)

            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            for key in train_metrics.keys():
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")
            for key in train_metrics.keys():
                self.logger.info(
                    f"Train {key} : {train_metrics[key]} - Val {key} : {val_metrics[key]}"
                )

            if val_metrics[self.eval_metric] >= best_val_metric:
                self.logger.info(
                    f"Validation {self.eval_metric} score improved from {best_val_metric} to {val_metrics[self.eval_metric]}"
                )
                best_val_metric = val_metrics[self.eval_metric]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_folder, "best_model.pt"),
                )
            else:
                self.logger.info(
                    f"Validation {self.eval_metric} score didn't improve from {best_val_metric}"
                )

        end = time.time()
        time_taken = end - start
        self.logger.info(
            "Training completed in {:.0f}h {:.0f}m {:.0f}s".format(
                time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60
            )
        )
        self.logger.info(f"Best Val {self.eval_metric} score: {best_val_metric}")

        return history

    def calc_loss(self, y_pred, y_true):
        """
        Crossentropy loss for predicted tokens

        :param y_pred: Predicted tokens
        :type y_pred: torch.Tensor (batch_size, seq_len, num_vocab)
        :param y_true: True tokens
        :type y_true: torch.Tensor (batch_size, seq_len)
        :return: BCE Loss
        :rtype: torch.float32
        """
        y_pred = torch.reshape(y_pred, (-1, 2))
        y_true = torch.reshape(y_true, (-1, 2))

        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        return loss_fn(y_pred, y_true)
