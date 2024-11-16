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

from metrics import TextGenerationMetrics


class MultiHeadAttention(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(MultiHeadAttention, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.d_model = config_dict["model"]["d_model"]
        self.num_heads = config_dict["model"]["num_heads"]
        self.d_qkv = self.d_model // self.num_heads

        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wo = nn.Linear(self.d_qkv * self.num_heads, self.d_model)

    def forward(self, Q, K, V, mask=False):
        """
        _summary_

        :param Q: _description_
        :type Q: _type_
        :param K: _description_
        :type K: _type_
        :param V: _description_
        :type V: _type_
        :param mask: _description_, defaults to False
        :type mask: bool, optional
        :return: _description_
        :rtype: _type_
        """
        batch_size = Q.size(0)
        Q = Q.view(batch_size, self.seq_len, self.num_heads, self.d_qkv).transpose(1, 2)
        K = K.view(batch_size, self.seq_len, self.num_heads, self.d_qkv).transpose(1, 2)
        V = V.view(batch_size, self.seq_len, self.num_heads, self.d_qkv).transpose(1, 2)

        attn_qkv = self._scaled_dotproduct_attention(Q, K, V, mask=mask)
        attn_qkv = attn_qkv.transpose(1, 2).reshape(
            batch_size, self.seq_len, self.d_model
        )

        attn_qkv = self.Wo(attn_qkv)

        return attn_qkv

    def _scaled_dotproduct_attention(self, Q, K, V, mask=None):
        """
        _summary_

        :param Q: _description_
        :type Q: _type_
        :param K: _description_
        :type K: _type_
        :param V: _description_
        :type V: _type_
        :param mask: _description_, defaults to None
        :type mask: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        matmul = torch.matmul(Q, K.transpose(-1, -2))
        if mask:
            mask_ids = torch.triu_indices(self.seq_len, self.seq_len)
            matmul[:, :, mask_ids[0], mask_ids[1]] = -1e9
        scale = matmul / math.sqrt(self.d_qkv)
        softmax = nn.Softmax(dim=-1)(scale)
        attn_qkv = torch.matmul(softmax, V)

        return attn_qkv


class PositionalEncoding(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(PositionalEncoding, self).__init__()

        d_model = config_dict["model"]["d_model"]
        seq_len = config_dict["dataset"]["seq_len"]

        self.pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1)
        denom = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        self.pe[:, 0::2] = torch.sin(pos / denom)
        self.pe[:, 1::2] = torch.cos(pos / denom)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        """
        _summary_

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        x = x + self.pe

        return x


class FeedForward(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(FeedForward, self).__init__()

        d_model = config_dict["model"]["d_model"]
        d_ff = config_dict["model"]["d_ff"]

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        _summary_

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)

        return x


class EncoderLayer(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(EncoderLayer, self).__init__()
        dropout = config_dict["model"]["dropout"]
        d_model = config_dict["model"]["d_model"]

        self.mh_self_attn = MultiHeadAttention(config_dict)

        self.feed_forward = FeedForward(config_dict)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """
        _summary_

        :param src: _description_
        :type src: _type_
        :return: _description_
        :rtype: _type_
        """
        attn_output = self.mh_self_attn(src, src, src)
        output = self.layer_norm1(src + self.dropout1(attn_output))
        ffwd_output = self.feed_forward(output)
        output = self.layer_norm2(output + self.dropout2(ffwd_output))

        return output


class DecoderLayer(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(DecoderLayer, self).__init__()
        dropout = config_dict["model"]["dropout"]
        d_model = config_dict["model"]["d_model"]

        self.mh_masked_self_attn = MultiHeadAttention(config_dict)
        self.mh_cross_attn = MultiHeadAttention(config_dict)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.feed_forward = FeedForward(config_dict)

    def forward(self, enc_output, tgt):
        """
        _summary_

        :param enc_output: _description_
        :type enc_output: _type_
        :param tgt: _description_
        :type tgt: _type_
        :return: _description_
        :rtype: _type_
        """
        masked_attn_output = self.mh_masked_self_attn(tgt, tgt, tgt, True)
        output = self.layer_norm1(tgt + self.dropout1(masked_attn_output))

        cross_attn_output = self.mh_cross_attn(enc_output, enc_output, output)
        output = self.layer_norm2(cross_attn_output + self.dropout2(output))

        ffwd_output = self.feed_forward(output)
        output = self.layer_norm3(output + self.dropout3(ffwd_output))

        return output


class TransformerModel(nn.Module):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        super(TransformerModel, self).__init__()

        embed_dim = config_dict["model"]["d_model"]
        num_vocab = config_dict["dataset"]["num_vocab"]
        num_layers = config_dict["model"]["num_layers"]
        dropout = config_dict["model"]["dropout"]

        self.src_embed_layer = nn.Embedding(num_vocab, embed_dim)
        self.tgt_embed_layer = nn.Embedding(num_vocab, embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.positional_encoding = PositionalEncoding(config_dict)

        self.encoder_layers = [EncoderLayer(config_dict) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(config_dict) for _ in range(num_layers)]

        self.classifier_layer = nn.Linear(embed_dim, num_vocab)

    def forward(self, src, tgt):
        """
        _summary_

        :param src: _description_
        :type src: _type_
        :param tgt: _description_
        :type tgt: _type_
        :return: _description_
        :rtype: _type_
        """
        src_embed = self.dropout1(self.positional_encoding(self.src_embed_layer(src)))
        tgt_embed = self.dropout2(self.positional_encoding(self.tgt_embed_layer(tgt)))

        enc_output = src_embed
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        dec_output = tgt_embed
        for layer in self.decoder_layers:
            dec_output = layer(enc_output, dec_output)

        output = self.classifier_layer(dec_output)
        # output = nn.Softmax(dim=-1)(output)

        return output


class TransformerTrainer(nn.Module):
    """
    _summary_

    :param model: _description_
    :type model: _type_
    :param optimizer: _description_
    :type optimizer: _type_
    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, model, optimizer, config_dict):
        super(TransformerTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict
        self.metric_cls = TextGenerationMetrics(config_dict)
        self.eval_metric = config_dict["train"]["eval_metric"]

    def train_one_epoch(self, data_loader, epoch):
        """
        _summary_

        :param data_loader: _description_
        :type data_loader: _type_
        :param epoch: _description_
        :type epoch: _type_
        :return: _description_
        :rtype: _type_
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
            tgt_hat = self.model(src, tgt)

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
        _summary_

        :param data_loader: _description_
        :type data_loader: _type_
        :return: _description_
        :rtype: _type_
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
            tgt_hat = self.model(src, tgt)

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
        _summary_

        :param data_loader: _description_
        :type data_loader: _type_
        :return: _description_
        :rtype: _type_
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
            tgt_hat = self.model(src, tgt)

            y_pred.append(tgt_hat.cpu().detach().numpy())
            sents.append(sent[0].cpu().detach().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        sents = np.concatenate(sents, axis=0)

        return sents, y_pred

    def fit(self, train_loader, val_loader):
        """
        _summary_

        :param train_loader: _description_
        :type train_loader: _type_
        :param val_loader: _description_
        :type val_loader: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param y_pred: _description_
        :type y_pred: _type_
        :param y_true: _description_
        :type y_true: _type_
        :return: _description_
        :rtype: _type_
        """
        y_pred = torch.flatten(y_pred, end_dim=1)

        y_true = torch.flatten(y_true)
        y_true = F.one_hot(
            y_true, num_classes=self.config_dict["dataset"]["num_vocab"]
        ).to(torch.float)

        loss_fn = nn.CrossEntropyLoss(reduce="sum")

        return loss_fn(y_pred, y_true)
