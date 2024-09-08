import os
import tqdm
import time
import logging
import numpy as np
from collections import defaultdict

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn


### RNN Cell
class RNNCell(nn.Module):
    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(RNNCell, self).__init__()
        
        self.hh_dense = nn.Linear(h_dim, h_dim)
        self.hx_dense = nn.Linear(inp_x_dim, h_dim)

        self.xh_dense = nn.Linear(h_dim, out_x_dim)

    def forward(self, ht_1, xt):
        ht = nn.Tanh()(self.hh_dense(ht_1) + self.hx_dense(xt))
        yt = self.xh_dense(ht)
        return ht, yt

### Stacked RNN
class RNNModel(nn.Module):
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
            inp_x_dim, out_x_dim = self.x_dims[i], self.x_dims[i+1]
            self.rnn_cells.append(RNNCell(h_dim, inp_x_dim, out_x_dim))

    def forward(self, X):
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
        for i in range(len(self.clf_dims)-1):
            out = nn.Linear(self.clf_dims[i], self.clf_dims[i+1])(out)
            out = nn.ReLU()(out)

        out = nn.Linear(self.clf_dims[-1], self.num_classes)(out)

        return out

    def init_hidden(self):
        hts = [nn.init.kaiming_uniform_(torch.empty(self.num_samples, dim)) for dim in self.h_dims]

        return hts
    

class RNNTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        super(RNNTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.config_dict = config_dict

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        self.logger.info(f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------")
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(axis=1))
            y_pred.append(y_hat.cpu().detach().numpy().argmax(axis=1))

        train_loss = total_loss/num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        train_acc = accuracy_score(y_true, y_pred)

        return train_loss, train_acc

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        self.model.eval()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation")

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            loss = self.loss_fn(y, y_hat)

            total_loss += loss
            num_instances += y.size(0)

            y_true.append(y.cpu().detach().numpy().argmax(axis=1))
            y_pred.append(y_hat.cpu().detach().numpy().argmax(axis=1))

        val_loss = total_loss/num_instances
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_acc = accuracy_score(y_true, y_pred)

        return val_loss, val_acc
    
    @torch.no_grad()
    def predict(self, data_loader):
        self.model.eval()
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference")

        for batch_id, (X, y) in pbar:
            y_hat = self.model(X)

            y_true.append(y.cpu().detach().numpy())
            y_pred.append(y_hat.cpu().detach().numpy())
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        return y_true, y_pred

    def fit(self, train_loader, val_loader):
        num_epochs = self.config_dict["train"]["epochs"]
        output_folder = self.config_dict["paths"]["output_folder"]

        best_val_acc = -np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs+1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.val_one_epoch(val_loader)

            self.logger.info(f"Train Loss : {train_loss} - Train Acc: {train_acc} - Val Loss : {val_loss} - Val Acc: {val_acc}")
                
            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc >= best_val_acc:
                self.logger.info(f"Validation Accuracy improved from {best_val_acc} to {val_acc}")
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(output_folder, "best_model.pt"))
            else:
                self.logger.info(f"Validation Accuracy didn't improve from {best_val_acc}")

        end = time.time()
        time_taken = end-start
        self.logger.info('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60))
        self.logger.info(f"Best Val Accuracy: {best_val_acc}")

        return history

    