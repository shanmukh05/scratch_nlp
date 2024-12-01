import os
import tqdm
import time
import logging
import numpy as np
from collections import defaultdict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import TextGenerationMetrics


### LSTM Cell
class LSTMCell(nn.Module):
    """
    LSTM Cell

    :param h_dim: Hidden state vector dimension
    :type h_dim: int
    :param inp_x_dim: Input vector dimension
    :type inp_x_dim: int
    :param out_x_dim: Output vector dimension
    :type out_x_dim: int
    """

    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(LSTMCell, self).__init__()

        self.wf_dense = nn.Linear(h_dim, h_dim)
        self.uf_dense = nn.Linear(inp_x_dim, h_dim)

        self.wi_dense = nn.Linear(h_dim, h_dim)
        self.ui_dense = nn.Linear(inp_x_dim, h_dim)

        self.wo_dense = nn.Linear(h_dim, h_dim)
        self.uo_dense = nn.Linear(inp_x_dim, h_dim)

        self.wc_dense = nn.Linear(h_dim, h_dim)
        self.uc_dense = nn.Linear(inp_x_dim, h_dim)

        self.xh_dense = nn.Linear(h_dim, out_x_dim)

    def forward(self, ht_1, ct_1, xt):
        """
        Forward propogation

        :param ht_1: Hidden state vector
        :type ht_1: torch.Tensor (batch_size, h_dim)
        :param ct_1: Cell stae vector
        :type ct_1: torch.Tensor (batch_size, h_dim)
        :param xt: Input vector
        :type xt: torch.Tensor (batch_size, embed_dim)
        :return: New hidden, cell states, output
        :rtype: tuple (torch.Tensor [batch_size, h_dim], torch.Tensor [batch_size, h_dim], torch.Tensor [batch_size, out_dim])
        """
        ft = nn.Sigmoid()(self.wf_dense(ht_1) + self.uf_dense(xt))
        it = nn.Sigmoid()(self.wi_dense(ht_1) + self.ui_dense(xt))
        ot = nn.Sigmoid()(self.wo_dense(ht_1) + self.uo_dense(xt))

        ct_ = nn.Tanh()(self.wc_dense(ht_1) + self.uc_dense(xt))
        ct = ft * ct_1 + it * ct_
        ht = ot * nn.Tanh()(ct)

        yt = self.xh_dense(ht)

        return ht, ct, yt


### LSTM Model
class LSTMModel(nn.Module):
    """
    LSTM Architecture

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(LSTMModel, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.num_layers = config_dict["model"]["num_layers"]
        self.h_dims = config_dict["model"]["h_dim"]
        self.x_dims = config_dict["model"]["x_dim"]
        image_dim = config_dict["preprocess"]["image_dim"]

        num_vocab = config_dict["dataset"]["num_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        self.embed_layer = nn.Embedding(num_vocab, embed_dim)

        image_backbone = config_dict["model"]["image_backbone"]
        self.im_head = timm.create_model(image_backbone, pretrained=True, num_classes=0)
        im_feat_dim = self.im_head(torch.rand(tuple([1] + image_dim))).data.shape[-1]
        self.im_dense = nn.Linear(im_feat_dim, embed_dim)

        self.word_classifier = nn.Linear(self.x_dims[-1], num_vocab)

        self.lstm_cells = []
        for i in range(self.num_layers):
            h_dim = self.h_dims[i]
            inp_x_dim, out_x_dim = self.x_dims[i], self.x_dims[i + 1]
            self.lstm_cells.append(LSTMCell(h_dim, inp_x_dim, out_x_dim))

    def forward(self, images, tokens=None):
        """
        Forward propogation

        :param images: Images tensor
        :type images: torch.Tensor (batch_size, num_channels, h_dim, w_dim)
        :param tokens: Captions tokens, defaults to None
        :type tokens: torch.Tensor (batch_size, seq_len), optional
        :return: Predicted captions
        :rtype: torch.Tensor (batch_size, seq_len, num_vocab)
        """
        self.num_samples = images.size(0)

        hts = self.init_hidden()
        cts = self.init_hidden()

        im_feat = self.im_head(images)
        yt = self.im_dense(im_feat)

        for i in range(self.num_layers):
            ht, ct = hts[i], cts[i]
            ht, ct, yt = self.lstm_cells[i](ht, ct, yt)
            hts[i], cts[i] = ht, ct

        if tokens is not None:
            x_embed = self.embed_layer(tokens.to(torch.long))
            yt = x_embed[:, 0, :]
        else:
            start_token = torch.zeros(self.num_samples, 1).to(torch.long)
            yt = self.embed_layer(start_token)[:, 0, :]

        pts = []
        for j in range(self.seq_len):
            for i in range(self.num_layers):
                ht, ct = hts[i], cts[i]
                ht, ct, yt = self.lstm_cells[i](ht, ct, yt)
                hts[i], cts[i] = ht, ct

            yt = self.word_classifier(yt)
            # pt = nn.Softmax()(yt)[:, None, :]
            pt = yt[:, None, :]
            pts.append(pt)

            if j >= self.seq_len - 1:
                break
            if tokens is not None:
                yt = x_embed[:, j + 1, :]
            else:
                yt = yt.argmax(axis=1)
                yt = self.embed_layer(yt.to(torch.long))

        return torch.concat(pts, dim=1)

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


### LSTMTrainer
class LSTMTrainer(nn.Module):
    """
    LSTM Trainer

    :param model: LSTM model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, model, optimizer, config_dict):
        super(LSTMTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict
        self.metric_cls = TextGenerationMetrics(config_dict)
        self.eval_metric = config_dict["train"]["eval_metric"]

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

        for batch_id, (images, tokens) in pbar:
            tokens = tokens.to(torch.long)
            tokens_hat = self.model(images, tokens)

            loss = self.calc_loss(tokens_hat, tokens)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += tokens.size(0)

            y_true.append(tokens.cpu().detach().numpy())
            y_pred.append(tokens_hat.cpu().detach().numpy())

        train_loss = total_loss / num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        train_metrics = self.metric_cls.get_metrics(y_true, y_pred)

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

        for batch_id, (images, tokens) in pbar:
            tokens = tokens.to(torch.long)
            tokens_hat = self.model(images, tokens)

            loss = self.calc_loss(tokens_hat, tokens)

            total_loss += loss
            num_instances += tokens.size(0)

            y_true.append(tokens.cpu().detach().numpy())
            y_pred.append(tokens_hat.cpu().detach().numpy())

        val_loss = total_loss / num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_metrics = self.metric_cls.get_metrics(y_true, y_pred)

        return val_loss, val_metrics

    @torch.no_grad()
    def predict(self, data_loader):
        """
        Runs inference to predict a translation of soruce sentence

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: Predicted tokens
        :rtype: numpy.ndarray (num_samples, seq_len, num_vocab)
        """
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, images in pbar:
            tokens_hat = self.model(images)
            y_pred.append(tokens_hat.cpu().detach().numpy())

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

            if val_metrics[self.eval_metric] <= best_val_metric:
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
        y_pred = torch.flatten(y_pred, end_dim=1)

        y_true = torch.flatten(y_true)
        y_true = F.one_hot(
            y_true, num_classes=self.config_dict["dataset"]["num_vocab"]
        ).to(torch.float)

        loss_fn = nn.CrossEntropyLoss(reduce="sum")

        return loss_fn(y_pred, y_true)
