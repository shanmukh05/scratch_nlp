import os
import tqdm
import time
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn


class GloVeModel(nn.Module):
    """
    GloVe Model

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        super(GloVeModel, self).__init__()

        self.embed_dim = config_dict["model"]["embed_dim"]
        self.num_vocab = 1 + config_dict["dataset"]["num_vocab"]

        self.cxt_embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.ctr_embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.cxt_bias_embedding = nn.Embedding(self.num_vocab, 1)
        self.ctr_bias_embedding = nn.Embedding(self.num_vocab, 1)

    def forward(self, ctr, cxt):
        """
        Forward propogation

        :param ctr: Center tokens
        :type ctr: torch.Tensor (batch_size,)
        :param cxt: Context tokens
        :type cxt: torch.Tensor (batch_size,)
        :return: Center, Context Embeddings and Biases
        :rtype: tuple (torch.Tensor [batch_size, embed_dim], torch.Tensor [batch_size, embed_dim],torch.Tensor [batch_size, 1],torch.Tensor [batch_size, 1],)
        """
        ctr = ctr.to(dtype=torch.long)
        cxt = cxt.to(dtype=torch.long)

        ctr_embed = self.ctr_embedding(ctr)
        cxt_embed = self.cxt_embedding(cxt)

        ctr_bias = self.ctr_bias_embedding(ctr)
        cxt_bias = self.cxt_bias_embedding(cxt)

        return ctr_embed, cxt_embed, ctr_bias, cxt_bias


class GloVeTrainer(nn.Module):
    """
    GloVe Trainer

    :param model: Seq2Seq model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict 
    """
    def __init__(self, model, optimizer, config_dict):
        super(GloVeTrainer, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict

        self.x_max = config_dict["train"]["x_max"]
        self.alpha = config_dict["train"]["alpha"]

    def train_one_epoch(self, data_loader, epoch):
        """
        Train step

        :param data_loader: Train Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :param epoch: Epoch number
        :type epoch: int
        :return: Train Loss
        :rtype: torch.float32
        """
        self.model.train()
        total_loss, num_instances = 0, 0

        self.logger.info(
            f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------"
        )
        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
        )

        for batch_id, (ctr, cxt, cnt) in pbar:
            ctr_embed, cxt_embed, ctr_bias, cxt_bias = self.model(ctr, cxt)

            loss = self.loss_fn(ctr_embed, cxt_embed, ctr_bias, cxt_bias, cnt)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += cnt.size(0)

        train_loss = total_loss / num_instances

        return train_loss

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        """
        Validation step

        :param data_loader: Validation Data Loader
        :type data_loader: torch.utils.data.Dataloader
        :return: Validation Loss
        :rtype: torch.float32
        """
        self.model.eval()
        total_loss, num_instances = 0, 0

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Validation"
        )

        for batch_id, (ctr, cxt, cnt) in pbar:
            ctr_embed, cxt_embed, ctr_bias, cxt_bias = self.model(ctr, cxt)

            loss = self.loss_fn(ctr_embed, cxt_embed, ctr_bias, cxt_bias, cnt)

            total_loss += loss
            num_instances += cnt.size(0)

        val_loss = total_loss / num_instances

        return val_loss

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
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.val_one_epoch(val_loader)

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")

            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))

            if val_loss <= best_val_loss:
                self.logger.info(
                    f"Validation Loss improved from {best_val_loss} to {val_loss}"
                )
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_folder, "best_model.pt"),
                )
            else:
                self.logger.info(f"Validation loss didn't improve from {best_val_loss}")

        end = time.time()
        time_taken = end - start
        self.logger.info(
            "Training completed in {:.0f}h {:.0f}m {:.0f}s".format(
                time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60
            )
        )
        self.logger.info(f"Best Val RMSE: {best_val_loss}")

        return history

    def loss_fn(self, ctr_embed, cxt_embed, ctr_bias, cxt_bias, count):
        """
        GloVe loss

        :param ctr_embed: Center embedding
        :type ctr_embed: torch.Tensor (batch_size, embed_dim)
        :param cxt_embed: Context embedding
        :type cxt_embed: torch.Tensor (batch_size, embed_dim)
        :param ctr_bias: Center Bias
        :type ctr_bias: torch.Tensor (batch_size, 1)
        :param cxt_bias: Context Bias
        :type cxt_bias: torch.Tensor (batch_size, 1)
        :param count: Cooccurence matrix element for (center, context)
        :type count: float
        :return: Loss
        :rtype: torch.float32
        """
        factor = torch.pow(count / self.x_max, self.alpha)
        factor[factor > 1] = 1
        log_count = torch.log(1 + count)

        embed_product = torch.sum(ctr_embed * cxt_embed, dim=1)

        loss = factor * torch.pow(embed_product + ctr_bias + cxt_bias - log_count, 2)
        return torch.sum(loss)
