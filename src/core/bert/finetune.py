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


class BERTFinetuneTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        """
        _summary_

        :param model: _description_
        :type model: _type_
        :param optimizer: _description_
        :type optimizer: _type_
        :param config_dict: _description_
        :type config_dict: _type_
        """
        super(BERTFinetuneTrainer, self).__init__()
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
        total_loss, total_start_loss, total_end_loss, num_instances = 0, 0, 0, 0

        self.logger.info(
            f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------"
        )
        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Training"
        )

        for batch_id, (tokens, start_ids, end_ids) in pbar:
            tokens = tokens.to(torch.long)
            start_ids = start_ids.to(torch.long)
            end_ids = end_ids.to(torch.long)
            _, start_ids_prob, end_ids_prob = self.model(tokens)

            start_loss, end_loss = self.calc_loss(
                start_ids_prob, end_ids_prob, start_ids, end_ids
            )
            loss = start_loss + end_loss
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            total_start_loss += start_loss
            total_end_loss += end_loss
            num_instances += tokens.size(0)

        train_loss = total_loss / num_instances
        train_start_loss = total_start_loss / num_instances
        train_end_loss = total_end_loss / num_instances

        return train_loss, train_start_loss, train_end_loss

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
        total_loss, total_start_loss, total_end_loss, num_instances = 0, 0, 0, 0

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Validation"
        )

        for batch_id, (tokens, start_ids, end_ids) in pbar:
            tokens = tokens.to(torch.long)
            start_ids = start_ids.to(torch.long)
            end_ids = end_ids.to(torch.long)
            _, start_ids_prob, end_ids_prob = self.model(tokens)

            start_loss, end_loss = self.calc_loss(
                start_ids_prob, end_ids_prob, start_ids, end_ids
            )
            loss = start_loss + end_loss

            total_loss += loss
            total_start_loss += start_loss
            total_end_loss += end_loss
            num_instances += tokens.size(0)

        val_loss = total_loss / num_instances
        val_start_loss = total_start_loss / num_instances
        val_end_loss = total_end_loss / num_instances

        return val_loss, val_start_loss, val_end_loss

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
        y_start_ids, y_end_ids = [], []
        enc_outputs = []

        pbar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc="Inference"
        )

        for batch_id, (tokens, start_ids, end_ids) in pbar:
            tokens = tokens.to(torch.long)
            start_ids = start_ids.to(torch.long)
            end_ids = end_ids.to(torch.long)
            enc_output, _, _ = self.model(tokens)

            y_start_ids.append(start_ids.cpu().detach().numpy())
            y_end_ids.append(end_ids.cpu().detach().numpy())
            enc_outputs.append(enc_output.cpu().detach().numpy())

        y_start_ids = np.concatenate(y_start_ids, axis=0)
        y_end_ids = np.concatenate(y_end_ids, axis=0)
        enc_outputs = np.concatenate(enc_outputs, axis=0)

        return y_start_ids, y_end_ids, enc_outputs

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
        for epoch in range(1, num_epochs + 1):
            train_loss, train_start_loss, train_end_loss = self.train_one_epoch(
                train_loader, epoch
            )
            val_loss, val_start_loss, val_end_loss = self.val_one_epoch(val_loader)

            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))
            history["train_start_loss"].append(float(train_start_loss.detach().numpy()))
            history["val_start_loss"].append(float(val_start_loss.detach().numpy()))
            history["train_end_loss"].append(float(train_end_loss.detach().numpy()))
            history["val_end_loss"].append(float(val_end_loss.detach().numpy()))

            self.logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")

            if val_loss <= best_val_loss:
                self.logger.info(
                    f"Validation Loss score improved from {best_val_loss} to {val_loss}"
                )
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_folder, "best_model_finetune.pt"),
                )
            else:
                self.logger.info(
                    f"Validation Loss score didn't improve from {best_val_loss}"
                )

        end = time.time()
        time_taken = end - start
        self.logger.info(
            "Training completed in {:.0f}h {:.0f}m {:.0f}s".format(
                time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60
            )
        )
        self.logger.info(f"Best Val Loss score: {best_val_loss}")

        return history

    def calc_loss(self, start_ids_prob, end_ids_prob, start_ids, end_ids):
        """
        _summary_

        :param start_ids_prob: _description_
        :type start_ids_prob: _type_
        :param end_ids_prob: _description_
        :type end_ids_prob: _type_
        :param start_ids: _description_
        :type start_ids: _type_
        :param end_ids: _description_
        :type end_ids: _type_
        :return: _description_
        :rtype: _type_
        """
        loss_fn = nn.NLLLoss()

        start_loss = loss_fn(start_ids_prob, start_ids)
        end_loss = loss_fn(end_ids_prob, end_ids)

        return start_loss, end_loss
