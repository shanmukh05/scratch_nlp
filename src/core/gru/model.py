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
    def __init__(self, h_dim, inp_x_dim, out_x_dim):
        super(GRUCell, self).__init__()

    def forward(self):
        pass


class GRUModel(nn.Module):
    def __init__(self, config_dict):
        super(GRUModel, self).__init__()

    def forward(self):
        pass

    def init_hidden(self):
        pass

class GRUTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        super(GRUTrainer, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.metric_cls = ClassificationMetrics(config_dict)
        self.eval_metric = config_dict["train"]["eval_metric"]

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        self.logger.info(f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------")
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

        for batch_id, (images, tokens) in pbar:
            tokens = tokens.to(torch.long)
            tokens_hat = self.model(images, tokens)

            loss = self.loss_fn(tokens_hat, tokens)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += tokens.size(0)

            y_true.append(tokens.cpu().detach().numpy())
            y_pred.append(tokens_hat.cpu().detach().numpy())

        train_loss = total_loss/num_instances

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        train_metrics = self.metric_cls.get_metrics(y_true, y_pred)

        return train_loss, train_metrics

    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        self.model.eval()
        total_loss, num_instances = 0, 0
        y_true, y_pred = [], []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation")

        for batch_id, (images, tokens) in pbar:
            tokens = tokens.to(torch.long)
            tokens_hat = self.model(images, tokens)

            loss = self.loss_fn(tokens_hat, tokens)

            total_loss += loss
            num_instances += tokens.size(0)

            y_true.append(tokens.cpu().detach().numpy())
            y_pred.append(tokens_hat.cpu().detach().numpy())

        val_loss = total_loss/num_instances
        
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        val_metrics = self.metric_cls.get_metrics(y_true, y_pred)

        return val_loss, val_metrics
    
    @torch.no_grad()
    def predict(self, data_loader):
        self.model.eval()
        y_pred = []

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference")

        for batch_id, images in pbar:
            tokens_hat = self.model(images)
            y_pred.append(tokens_hat.cpu().detach().numpy())
        
        y_pred = np.concatenate(y_pred, axis=0)

        return y_pred

    def fit(self, train_loader, val_loader):
        num_epochs = self.config_dict["train"]["epochs"]
        output_folder = self.config_dict["paths"]["output_folder"]

        best_val_metric = np.inf
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

            if val_metrics[self.eval_metric] <= best_val_metric:
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

