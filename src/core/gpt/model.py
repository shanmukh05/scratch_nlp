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

from core.transformer.model import MultiHeadAttention, PositionalEncoding, FeedForward
from metrics import TextGenerationMetrics


class DecoderLayer(nn.Module):
    """
    GPT Decoder layer

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(DecoderLayer, self).__init__()

        dropout = config_dict["model"]["dropout"]
        d_model = config_dict["model"]["d_model"]

        self.mh_masked_self_attn = MultiHeadAttention(config_dict)
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(config_dict)

    def forward(self, tokens):
        """
        Forward propogation

        :param tokens: Input tokens
        :type tokens: torch.Tensor (num_samples, seq_len)
        :return: Decoder output
        :rtype: torch.Tensor (num_samples, seq_len, d_ff)
        """
        tokens = self.layer_norm(tokens)
        masked_attn_output = self.mh_masked_self_attn(tokens, tokens, tokens, True)
        output = tokens + self.dropout1(masked_attn_output)

        ffwd_output = self.feed_forward(output)
        output = output + self.dropout2(ffwd_output)

        return output


class GPTModel(nn.Module):
    """
    GPT Architecture

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(GPTModel, self).__init__()

        embed_dim = config_dict["model"]["d_model"]
        num_vocab = config_dict["dataset"]["num_vocab"]
        num_layers = config_dict["model"]["num_layers"]
        dropout = config_dict["model"]["dropout"]
        d_model = config_dict["model"]["d_model"]
        self.num_predict_tokens = config_dict["test"]["predict_tokens"]
        self.seq_len = config_dict["dataset"]["seq_len"]

        self.embed_layer = nn.Embedding(num_vocab, embed_dim)
        self.positional_encoding = PositionalEncoding(config_dict)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.decoder_layers = [DecoderLayer(config_dict) for _ in range(num_layers)]

        self.classifier_layer = nn.Linear(embed_dim, num_vocab)

    def forward(self, tokens):
        """
        Forward propogation

        :param tokens: Input tokens
        :type tokens: torch.Tensor (num_samples, seq_len)
        :return: probability of Generated Tokens
        :rtype: torch.Tensor (num_samples, seq_len, num_vocab)
        """
        embeds = self.dropout(self.positional_encoding(self.embed_layer(tokens)))

        dec_output = embeds
        for layer in self.decoder_layers:
            dec_output = layer(dec_output)

        output = self.layer_norm(dec_output)
        output = self.classifier_layer(output)

        return output

    def generate(self, tokens):
        """
        Generate Tokens

        :param tokens: Input tokens
        :type tokens: torch.Tensor (num_samples, seq_len + num_pred_tokens)
        :return: Generated tokens
        :rtype: torch.Tensor (num_samples, seq_len + num_pred_tokens)
        """
        tokens_pred = torch.zeros_like(tokens)
        tokens_pred[:, : self.seq_len] = tokens[:, : self.seq_len]

        for i in range(self.num_predict_tokens):
            inputs = tokens_pred[:, i : i + self.seq_len]
            outputs = self.forward(inputs)
            new_token = torch.argmax(outputs[:, -1, :], dim=-1)
            tokens_pred[:, self.seq_len + i] = new_token.squeeze()

        return tokens_pred


class GPTTrainer(nn.Module):
    """
    GPT Trainer

    :param model: GPT model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, model, optimizer, config_dict):
        super(GPTTrainer, self).__init__()
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

        for batch_id, sent in pbar:
            src, tgt = sent[0][:, :-1], sent[0][:, 1:]
            src = src.to(torch.long)
            tgt = tgt.to(torch.long)
            tgt_hat = self.model(src)

            loss = self.calc_loss(tgt_hat, tgt)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += tgt.size(0)

            y_true.append(tgt.cpu().detach().numpy())
            y_pred.append(tgt_hat.cpu().detach().numpy())

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

        for batch_id, sent in pbar:
            src, tgt = sent[0][:, :-1], sent[0][:, 1:]
            src = src.to(torch.long)
            tgt = tgt.to(torch.long)
            tgt_hat = self.model(src)

            loss = self.calc_loss(tgt_hat, tgt)

            total_loss += loss
            num_instances += tgt.size(0)

            y_true.append(tgt.cpu().detach().numpy())
            y_pred.append(tgt_hat.cpu().detach().numpy())

        val_loss = total_loss / num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_metrics = self.metric_cls.get_metrics(y_true, y_pred)

        return val_loss, val_metrics

    @torch.no_grad()
    def predict(self, data_loader):
        """
        Runs inference to predict a shifted sentence

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: True tokens, Predicted tokens
        :rtype: tuple (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, seq_len, num_vocab])
        """
        self.model.eval()
        y_pred, sents = [], []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, sent in pbar:
            src, tgt = sent[0][:, :-1], sent[0][:, 1:]
            src = src.to(torch.long)
            tgt = tgt.to(torch.long)
            tgt_hat = self.model(src)

            y_pred.append(tgt_hat.cpu().detach().numpy())
            sents.append(sent[0].cpu().detach().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        sents = np.concatenate(sents, axis=0)

        return sents, y_pred

    @torch.no_grad()
    def generate(self, data_loader):
        """
        Runs inference to generate new text

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: True tokens, Generated tokens
        :rtype: tuple (numpy.ndarray [num_samples, seq_len + num_pred_tokens], numpy.ndarray [num_samples, seq_len + num_pred_tokens])
        """
        self.model.eval()
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Generation"
        )
        for batch_id, tokens in pbar:
            tokens = tokens[0].to(torch.long)
            tokens_pred = self.model.generate(tokens)

            y_pred.append(tokens_pred.cpu().detach().numpy())
            y_true.append(tokens.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        return y_true, y_pred

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
