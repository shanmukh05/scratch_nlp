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


class BERTPretrainTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        super(BERTPretrainTrainer, self).__init__()
        """
        _summary_
        """        
        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict

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
        total_loss, total_clf_loss, total_nsp_loss, num_instances = 0, 0, 0, 0

        self.logger.info(f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------")
        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

        for batch_id, (tokens, tokens_mask, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)
            
            clf_loss, nsp_loss = self.calc_loss(tokens_pred, nsp_pred, tokens, nsp_labels, tokens_mask)
            loss = clf_loss + nsp_loss
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            total_clf_loss += clf_loss
            total_nsp_loss += nsp_loss
            num_instances += tokens.size(0)

        train_loss = total_loss/num_instances
        train_clf_loss = total_clf_loss/num_instances
        train_nsp_loss = total_nsp_loss/num_instances

        return train_loss, train_clf_loss, train_nsp_loss

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
        total_loss, total_clf_loss, total_nsp_loss, num_instances = 0, 0, 0, 0

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Validation")

        for batch_id, (tokens, tokens_mask, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)
            
            clf_loss, nsp_loss = self.calc_loss(tokens_pred, nsp_pred, tokens, nsp_labels, tokens_mask)
            loss = clf_loss + nsp_loss
            
            total_loss += loss
            total_clf_loss += clf_loss
            total_nsp_loss += nsp_loss
            num_instances += tokens.size(0)

        val_loss = total_loss/num_instances
        val_clf_loss = total_clf_loss/num_instances
        val_nsp_loss = total_nsp_loss/num_instances

        return val_loss, val_clf_loss, val_nsp_loss
    
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
        y_tokens_pred, y_tokens_true = [], []
        y_nsp_pred, y_nsp_true = [], []
        

        pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference")

        for batch_id, (tokens, _, nsp_labels) in pbar:
            tokens_pred, nsp_pred = self.model(tokens)

            y_tokens_pred.append(tokens_pred.cpu().detach().numpy())
            y_tokens_true.append(tokens.cpu().detach().numpy())

            y_nsp_true.append(nsp_labels.cpu().detach().numpy())
            y_nsp_pred.append(nsp_pred.cpu().detach().numpy())
        
        y_tokens_pred = np.concatenate(y_tokens_pred, axis=0)
        y_tokens_true = np.concatenate(y_tokens_true, axis=0)
        y_nsp_pred = np.concatenate(y_nsp_pred, axis=0)
        y_nsp_true = np.concatenate(y_nsp_true, axis=0)

        return y_tokens_true, y_nsp_true, y_tokens_pred, y_nsp_pred

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

        best_val_loss = np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs+1):
            train_loss, train_clf_loss, train_nsp_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_clf_loss, val_nsp_loss = self.val_one_epoch(val_loader)
                
            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            history["train_clf_loss"].append(float(train_clf_loss.detach().numpy()))
            history["val_clf_loss"].append(float(val_clf_loss.detach().numpy()))
            history["train_nsp_loss"].append(float(train_nsp_loss.detach().numpy()))
            history["val_nsp_loss"].append(float(val_nsp_loss.detach().numpy()))

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")

            if val_loss <= best_val_loss:
                self.logger.info(f"Validation Loss score improved from {best_val_loss} to {val_loss}")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(output_folder, "best_model_pretrain.pt"))
            else:
                self.logger.info(f"Validation Loss score didn't improve from {best_val_loss}")

        end = time.time()
        time_taken = end-start
        self.logger.info('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60))
        self.logger.info(f"Best Val Loss score: {best_val_loss}")

        return history
    
    def calc_loss(self, tokens_pred, nsp_pred, tokens_true, nsp_labels, tokens_mask):
        """
        _summary_

        :param tokens_pred: _description_
        :type tokens_pred: _type_
        :param nsp_pred: _description_
        :type nsp_pred: _type_
        :param tokens_true: _description_
        :type tokens_true: _type_
        :param nsp_labels: _description_
        :type nsp_labels: _type_
        :param tokens_mask: _description_
        :type tokens_mask: _type_
        :return: _description_
        :rtype: _type_
        """        
        num_vocab = self.config_dict["dataset"]["num_vocab"]
        nsp_loss_fn = nn.BCELoss()
        nsp_loss = nsp_loss_fn(nsp_pred.squeeze(), nsp_labels)

        mask_flatten = torch.flatten(tokens_mask).bool()
        mask_flatten = torch.stack([mask_flatten]*num_vocab, dim=1)

        y_pred = torch.flatten(tokens_pred, end_dim=1)
        y_true = torch.flatten(tokens_true)
        y_true = F.one_hot(
                    y_true,
                    num_classes = num_vocab
                ).to(torch.float)

        y_pred_mask = torch.masked_select(y_pred, mask_flatten).reshape(-1, y_pred.shape[1])
        y_true_mask = torch.masked_select(y_true, mask_flatten).reshape(-1, y_true.shape[1])

        clf_loss_fn = nn.CrossEntropyLoss(reduce="sum")
        clf_loss = clf_loss_fn(y_pred_mask, y_true_mask)     

        return clf_loss, nsp_loss