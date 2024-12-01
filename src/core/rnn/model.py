import os
import tqdm
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from metrics import ClassificationMetrics


### RNN Cell
class RNNCell(nn.Module):
    """
    RNN Cell

    :param h_dim: Hidden state vector dimension
    :type h_dim: int
    :param inp_x_dim: Input vector dimension
    :type inp_x_dim: int
    :param out_x_dim: Output vector dimension
    :type out_x_dim: int
    """
    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(RNNCell, self).__init__()

        self.hh_dense = nn.Linear(h_dim, h_dim)
        self.hx_dense = nn.Linear(inp_x_dim, h_dim)

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
        ht = nn.Tanh()(self.hh_dense(ht_1) + self.hx_dense(xt))
        yt = self.xh_dense(ht)
        return ht, yt


### Stacked RNN
class RNNModel(nn.Module):
    """
    RNN Architecture

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        super(RNNModel, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.num_layers = config_dict["model"]["num_layers"]
        self.h_dims = config_dict["model"]["h_dim"]
        self.x_dims = config_dict["model"]["x_dim"]
        self.clf_dims = config_dict["model"]["clf_dim"]
        self.num_classes = config_dict["dataset"]["num_classes"]

        num_vocab = 2 + config_dict["dataset"]["num_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        self.embed_layer = nn.Embedding(num_vocab, embed_dim)

        self.rnn_cells = []
        for i in range(self.num_layers):
            h_dim = self.h_dims[i]
            inp_x_dim, out_x_dim = self.x_dims[i], self.x_dims[i + 1]
            self.rnn_cells.append(RNNCell(h_dim, inp_x_dim, out_x_dim))

    def forward(self, X):
        """
        Forward propogation

        :param X: Input tokens
        :type X: torch.Tensor (batch_size, seq_len)
        :return: Prediction labels
        :rtype: torch.Tensor (batch_size, num_classes)
        """
        x_embed = self.embed_layer(X.to(torch.long))
        self.num_samples = x_embed.size(0)

        yt_1s = x_embed
        hts = self.init_hidden()
        for i in range(self.num_layers):
            yts = []
            rnn_cell = self.rnn_cells[i]

            ht = hts[i]
            for j in range(self.seq_len):
                ht, yt = rnn_cell(ht, yt_1s[:, j, :])
                yts.append(yt)

            yt_1s = torch.transpose(torch.stack(yts), 0, 1)

        out = yt_1s[:, -1, :]
        for i in range(len(self.clf_dims) - 1):
            out = nn.Linear(self.clf_dims[i], self.clf_dims[i + 1])(out)
            out = nn.ReLU()(out)

        out = nn.Linear(self.clf_dims[-1], self.num_classes)(out)

        return out

    def init_hidden(self):
        """
        Initialized hidden states

        :return: List of hidden states
        :rtype: list
        """
        hts = [
            nn.init.kaiming_uniform_(torch.empty(self.num_samples, dim))
            for dim in self.h_dims
        ]

        return hts


class RNNTrainer(nn.Module):
    """
    RNN Trainer

    :param model: RNN model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict 
    """
    def __init__(self, model, optimizer, config_dict):
        super(RNNTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
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

            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(axis=1))
            y_pred.append(y_hat.cpu().detach().numpy())

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

            loss = self.loss_fn(y, y_hat)

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(axis=1))
            y_pred.append(y_hat.cpu().detach().numpy())

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
        :return: Predicted tokens
        :rtype: numpy.ndarray (num_samples, num_classes)
        """
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )
        for batch_id, X in pbar:
            y_hat = self.model(X[0])

            y_pred.append(y_hat.cpu().detach().numpy())

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

        best_val_metric = np.inf
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
