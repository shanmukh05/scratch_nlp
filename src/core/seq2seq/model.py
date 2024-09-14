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
        ft = nn.Sigmoid()(self.wf_dense(ht_1) + self.uf_dense(xt))
        it = nn.Sigmoid()(self.wi_dense(ht_1) + self.ui_dense(xt))
        ot = nn.Sigmoid()(self.wo_dense(ht_1) + self.uo_dense(xt))

        ct_ = nn.Tanh()(self.wc_dense(ht_1) + self.uc_dense(xt))
        ct = ft*ct_1 + it*ct_
        ht = ot*nn.Tanh()(ct)

        return ht, ct


class Seq2SeqEncoder(nn.Module):
    def __init__(self, config_dict):
        super(Seq2SeqEncoder, self).__init__()

        self.seq_len = config_dict["dataset"]["seq_len"]
        self.num_layers = config_dict["model"]["num_layers"]
        self.h_dims = config_dict["model"]["h_dim"]
        self.x_dims = config_dict["model"]["x_dim"]

        num_src_vocab = config_dict["dataset"]["num_src_vocab"]
        embed_dim = config_dict["model"]["embed_dim"]
        self.src_embed_layer = nn.Embedding(num_src_vocab, embed_dim)

        self.enc_fwd_lstm_cells, self.enc_bwd_lstm_cells = [], []
        self.enc_y_dense_layers = []
        for i in range(self.num_layers):
            h_dim = self.h_dims[i]
            inp_x_dim, out_x_dim = self.x_dims[i], self.x_dims[i+1]
            self.enc_fwd_lstm_cells.append(EncoderLSTMCell(h_dim, inp_x_dim))
            self.enc_bwd_lstm_cells.append(EncoderLSTMCell(h_dim, inp_x_dim))
            self.enc_y_dense_layers.append(nn.Linear(2*h_dim, out_x_dim))

    def forward(self, src):
        self.num_samples = src.size(0)

        hts_fwd, cts_fwd = self.init_hidden(), self.init_hidden()
        hts_bwd, cts_bwd = self.init_hidden(), self.init_hidden()

        src_embed = self.src_embed_layer(src.to(torch.long))
        yts = list(torch.transpose(src_embed, 1, 0))
        hts = defaultdict()

        for i in range(self.num_layers):
            hts_fwd_dict, hts_bwd_dict = defaultdict(), defaultdict()
            ht_fwd, ct_fwd = hts_fwd[i], cts_fwd[i]
            ht_bwd, ct_bwd = hts_bwd[i], cts_bwd[i]

            for j in range(self.seq_len):
                ht_fwd, ct_fwd = self.enc_fwd_lstm_cells[i](ht_fwd, ct_fwd, yts[j])
                hts_fwd_dict[j] = ht_fwd

                ht_bwd, ct_bwd = self.enc_bwd_lstm_cells[i](ht_bwd, ct_bwd, yts[self.seq_len-j-1])
                hts_bwd_dict[self.seq_len-j-1] = ht_bwd

            for w in range(self.seq_len):
                ht_fwd, ht_bwd = hts_fwd_dict[w], hts_bwd_dict[w]
                ht = torch.cat([ht_fwd, ht_bwd], dim=-1)
                hts[w] = ht

                yt = self.enc_y_dense_layers[i](ht)
                yts[w] = yt

        return list(hts.values())


    def init_hidden(self):
        hts = [nn.init.kaiming_uniform_(torch.empty(self.num_samples, dim)) for dim in self.h_dims]

        return hts

class Seq2SeqAttention(nn.Module):
    def __init__(self):
        pass

    def forward(self, ):
        pass

class Seq2SeqDecoder(nn.Module):
    def __init__(self, config_dict):
        pass

    def forward(self):
        pass


class Seq2SeqModel(nn.Module):
    def __init__(self, config_dict):
        pass

    def forward(self):
        pass


class Seq2SeqTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        super(Seq2SeqTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict
        self.metric_cls = TextGenerationMetrics(config_dict)
        self.eval_metric = config_dict["train"]["eval_metric"]
        self.target_names = list(config_dict["dataset"]["labels"])

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        self.logger.info(f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------")
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.calc_loss(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(y_hat.cpu().detach().numpy())

        train_loss = total_loss/num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        train_metrics = self.metric_cls.get_metrics(y_true, y_pred, self.target_names)

        return train_loss, train_metrics

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        self.model.eval()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation")

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.calc_loss(y_hat, y)

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(y_hat.cpu().detach().numpy())

        val_loss = total_loss/num_instances
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_metrics = self.metric_cls.get_metrics(y_true, y_pred, self.target_names)

        return val_loss, val_metrics
    
    @torch.no_grad()
    def predict(self, data_loader):
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference")

        for batch_id, X in pbar:
            tokens_hat = self.model(X[0])
            y_pred.append(tokens_hat.cpu().detach().numpy())
        
        y_pred = np.concatenate(y_pred, axis=0)

        return y_pred

    def fit(self, train_loader, val_loader):
        num_epochs = self.config_dict["train"]["epochs"]
        output_folder = self.config_dict["paths"]["output_folder"]

        best_val_metric = -np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs+1):
            train_loss, train_metrics = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.val_one_epoch(val_loader)
                
            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            for key in train_metrics.keys():
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")
            for key in train_metrics.keys():
                self.logger.info(f"Train {key} : {train_metrics[key]} - Val {key} : {val_metrics[key]}")

            if val_metrics[self.eval_metric] >= best_val_metric:
                self.logger.info(f"Validation {self.eval_metric} score improved from {best_val_metric} to {val_metrics[self.eval_metric]}")
                best_val_metric = val_metrics[self.eval_metric]
                torch.save(self.model.state_dict(), os.path.join(output_folder, "best_model.pt"))
            else:
                self.logger.info(f"Validation {self.eval_metric} score didn't improve from {best_val_metric}")

        end = time.time()
        time_taken = end-start
        self.logger.info('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60))
        self.logger.info(f"Best Val {self.eval_metric} score: {best_val_metric}")

        return history
    
    def calc_loss(self, y_pred, y_true):
        y_pred = torch.reshape(y_pred, (-1, 2))
        y_true = torch.reshape(y_true, (-1, 2))

        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        return loss_fn(y_pred, y_true)

