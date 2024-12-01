import os
import tqdm
import time
import logging
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import TextGenerationMetrics


class EncoderLSTMCell(nn.Module):
    """
    Encoder LSTM Cell

    :param h_dim: Hidden state vector dimension
    :type h_dim: int
    :param inp_x_dim: Input vector dimension
    :type inp_x_dim: int
    """

    def __init__(self, h_dim, inp_x_dim):
        super(EncoderLSTMCell, self).__init__()

        self.wf_dense = nn.Linear(h_dim, h_dim)
        self.uf_dense = nn.Linear(inp_x_dim, h_dim)

        self.wi_dense = nn.Linear(h_dim, h_dim)
        self.ui_dense = nn.Linear(inp_x_dim, h_dim)

        self.wo_dense = nn.Linear(h_dim, h_dim)
        self.uo_dense = nn.Linear(inp_x_dim, h_dim)

        self.wc_dense = nn.Linear(h_dim, h_dim)
        self.uc_dense = nn.Linear(inp_x_dim, h_dim)

    def forward(self, ht_1, ct_1, xt):
        """
        Forward propogation

        :param ht_1: Hidden state vector
        :type ht_1: torch.Tensor (batch_size, h_dim)
        :param ct_1: Cell stae vector
        :type ct_1: torch.Tensor (batch_size, h_dim)
        :param xt: Input vector
        :type xt: torch.Tensor (batch_size, embed_dim)
        :return: New hidden, cell states
        :rtype: tuple (torch.Tensor [batch_size, h_dim], torch.Tensor [batch_size, h_dim])
        """
        ft = nn.Sigmoid()(self.wf_dense(ht_1) + self.uf_dense(xt))
        it = nn.Sigmoid()(self.wi_dense(ht_1) + self.ui_dense(xt))
        ot = nn.Sigmoid()(self.wo_dense(ht_1) + self.uo_dense(xt))

        ct_ = nn.Tanh()(self.wc_dense(ht_1) + self.uc_dense(xt))
        ct = ft * ct_1 + it * ct_
        ht = ot * nn.Tanh()(ct)

        return ht, ct


### LSTM Cell
class DecoderLSTMCell(nn.Module):
    """
    Decode LSTM cell

    :param h_dim: Hidden state vector dimension
    :type h_dim: int
    :param inp_x_dim: Input vector dimension
    :type inp_x_dim: int
    :param out_x_dim: Output vector dimension
    :type out_x_dim: int
    """

    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(DecoderLSTMCell, self).__init__()

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


class Seq2SeqEncoder(nn.Module):
    """
    Seq2Seq Encoder

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(Seq2SeqEncoder, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.num_layers = config_dict["model"]["num_layers"]
        self.h_dims = config_dict["model"]["encoder_h_dim"]
        self.x_dims = config_dict["model"]["encoder_x_dim"]
        self.x_dims.append(2 * self.h_dims[-1])

        num_src_vocab = config_dict["dataset"]["num_src_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        self.src_embed_layer = nn.Embedding(num_src_vocab, embed_dim)

        self.enc_fwd_lstm_cells, self.enc_bwd_lstm_cells = [], []
        self.enc_y_dense_layers = []
        for i in range(self.num_layers):
            h_dim = self.h_dims[i]
            inp_x_dim, out_x_dim = self.x_dims[i], self.x_dims[i + 1]
            self.enc_fwd_lstm_cells.append(EncoderLSTMCell(h_dim, inp_x_dim))
            self.enc_bwd_lstm_cells.append(EncoderLSTMCell(h_dim, inp_x_dim))
            self.enc_y_dense_layers.append(nn.Linear(2 * h_dim, out_x_dim))

    def forward(self, src):
        """
        Forward propogation

        :param src: Source tokens
        :type src: torch.Tensor (batch_size, seq_len)
        :return: Predicted tokens, Hidden states
        :rtype: tuple (torch.Tensor [batch_size, seq_len, out_dim], torch.Tensor [batch_size, 2*h_dim])
        """
        self.num_samples = src.size(0)

        hts_fwd, cts_fwd = self.init_hidden(), self.init_hidden()
        hts_bwd, cts_bwd = self.init_hidden(), self.init_hidden()

        src_embed = self.src_embed_layer(src.to(torch.long))
        yts = list(torch.transpose(src_embed, 1, 0))

        for i in range(self.num_layers):
            hts_fwd_dict, hts_bwd_dict = defaultdict(), defaultdict()
            ht_fwd, ct_fwd = hts_fwd[i], cts_fwd[i]
            ht_bwd, ct_bwd = hts_bwd[i], cts_bwd[i]

            for j in range(self.seq_len):
                ht_fwd, ct_fwd = self.enc_fwd_lstm_cells[i](ht_fwd, ct_fwd, yts[j])
                hts_fwd_dict[j] = ht_fwd

                ht_bwd, ct_bwd = self.enc_bwd_lstm_cells[i](
                    ht_bwd, ct_bwd, yts[self.seq_len - j - 1]
                )
                hts_bwd_dict[self.seq_len - j - 1] = ht_bwd

            for w in range(self.seq_len):
                ht_fwd, ht_bwd = hts_fwd_dict[w], hts_bwd_dict[w]
                ht = torch.cat([ht_fwd, ht_bwd], dim=-1)

                yt = self.enc_y_dense_layers[i](ht)
                yts[w] = yt

        return torch.stack(yts, dim=1), ht

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


class Seq2SeqAttention(nn.Module):
    """
    Seq2Seq Attention layer

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(Seq2SeqAttention, self).__init__()

        decoder_s_dim = 2 * config_dict["model"]["encoder_h_dim"][-1]
        encoder_y_dim = config_dict["model"]["encoder_x_dim"][-1]

        self.si_dense = nn.Linear(decoder_s_dim, decoder_s_dim)
        self.yi_dense = nn.Linear(encoder_y_dim, decoder_s_dim)

        self.attn_weights = nn.Linear(decoder_s_dim, 1)

    def forward(self, si_1, yts):
        """
        Forward Propogation

        :param si_1: Hidden state vector of decoder layer
        :type si_1: torch.Tensor (batch_size, 2*h_dim)
        :param yts: Encoder output vectors
        :type yts: torch.tensor (batch_size, seq_len, out_dim)
        :return: Attention weights, New hidden state vector
        :rtype: tuple (torch.Tensor [batch_size, seq_len], torch.Tensor [batch_size, 2*h_dim])
        """
        eij = self.attn_weights(self.si_dense(si_1.unsqueeze(1)) + self.yi_dense(yts))
        eij = eij.squeeze(2).unsqueeze(1)

        weights = F.softmax(eij, dim=-1)
        si = torch.bmm(weights, yts).squeeze()

        return weights, si


class Seq2SeqDecoder(nn.Module):
    """
    Seq2Seq Decoder

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(Seq2SeqDecoder, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        s_dim = 2 * config_dict["model"]["encoder_h_dim"][-1]
        decoder_x_dim = config_dict["model"]["encoder_x_dim"][-1]
        decoder_y_dim = config_dict["model"]["decoder_y_dim"]

        num_tgt_vocab = config_dict["dataset"]["num_tgt_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        self.tgt_embed_layer = nn.Embedding(num_tgt_vocab, embed_dim)

        self.attn_layer = Seq2SeqAttention(config_dict)
        self.dec_lstm_cell = DecoderLSTMCell(s_dim, decoder_x_dim, decoder_y_dim)

        self.tgt_word_classifier = nn.Linear(decoder_y_dim, num_tgt_vocab)

    def forward(self, encoder_yts, encoder_h, tgt=None):
        """
        Forward propogation

        :param encoder_yts: Encoder Output vectors
        :type encoder_yts: torch.Tensor (batch_size, seq_len, out_dim)
        :param encoder_h: Encoder final hidden vectors
        :type encoder_h: torch.Tensor (batch_size, seq_len, 2*h_dim)
        :param tgt: Target vectors, defaults to None
        :type tgt: torch.Tensor (batch_size, seq_len), optional
        :return: Final predictions, Attention weights
        :rtype: tuple (torch.Tensor [batch_size, seq_len, num_tgt_vocab], list)
        """
        batch_size = encoder_yts.size(0)

        if tgt is None:
            sos_token = 2 * torch.ones(batch_size, 1).to(torch.long)
            yt = self.tgt_embed_layer(sos_token)[:, 0, :]
        else:
            tgt_embeds = self.tgt_embed_layer(tgt.to(torch.long))
            yt = tgt_embeds[:, 0, :]

        st = encoder_h
        pts = []
        attn_weights = []
        for i in range(self.seq_len):
            weights, ct = self.attn_layer(st, encoder_yts)

            st, ct, yt = self.dec_lstm_cell(st, ct, yt)
            yt = self.tgt_word_classifier(yt)
            # pt = nn.Softmax()(yt)[:, None, :]
            pt = yt[:, None, :]
            pts.append(pt)
            attn_weights.append(weights)

            if i >= self.seq_len - 1:
                break

            if tgt is not None:
                yt = tgt_embeds[:, i + 1, :]
            else:
                yt = yt.argmax(axis=1)
                yt = self.tgt_embed_layer(yt.to(torch.long))

        return torch.concat(pts, dim=1), attn_weights


class Seq2SeqModel(nn.Module):
    """
    Seq2Seq Model Architecture

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        super(Seq2SeqModel, self).__init__()

        self.encoder = Seq2SeqEncoder(config_dict)
        self.decoder = Seq2SeqDecoder(config_dict)

    def forward(self, src, tgt=None):
        """
        Forward propogation

        :param src: Source tokens
        :type src: torch.Tensor (batch_size, seq_len)
        :param tgt: _Target tokens, defaults to None
        :type tgt: torch.Tensor (batch_size, seq_len), optional
        :return: Final predictions, Attention weights
        :rtype: tuple (torch.Tensor [batch_size, seq_len, num_tgt_vocab], list)
        """
        encoder_yts, encoder_h = self.encoder(src)
        tgt_probs, attn_weights = self.decoder(encoder_yts, encoder_h, tgt)

        return tgt_probs, attn_weights


class Seq2SeqTrainer(nn.Module):
    """
    Seq2Seq Trainer

    :param model: Seq2Seq model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim
    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, model, optimizer, config_dict):
        super(Seq2SeqTrainer, self).__init__()
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
        :return: Train Loss, Train Metrics
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

        for batch_id, (src, tgt) in pbar:
            tgt = tgt.to(torch.long)
            tgt_hat, attn_weights = self.model(src, tgt)

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
        :return: Validation Loss, Validation Metrics
        :rtype: tuple (torch.float32, dict)
        """
        self.model.eval()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Validation"
        )

        for batch_id, (src, tgt) in pbar:
            tgt = tgt.to(torch.long)
            tgt_hat, attn_weights = self.model(src, tgt)

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
        Runs inference to predict a translation of soruce sentence

        :param data_loader: Infer Data loader
        :type data_loader: torch.utils.data.DataLoader
        :return: Predicted tokens
        :rtype: numpy.ndarray (num_samples, seq_len, num_tgt_vocab)
        """
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, src in pbar:
            tgt_hat, attn_weights = self.model(src[0])
            y_pred.append(tgt_hat.cpu().detach().numpy())

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
            y_true, num_classes=self.config_dict["dataset"]["num_tgt_vocab"]
        ).to(torch.float)

        loss_fn = nn.CrossEntropyLoss(reduce="sum")

        return loss_fn(y_pred, y_true)
