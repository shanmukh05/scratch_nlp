import os
import time
import torch
import tqdm
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class Word2VecModel(nn.Module):
    def __init__(self, config_dict):
        super(Word2VecModel, self).__init__()

        self.embed_dim = config_dict["model"]["embed_dim"]
        self.num_vocab = 1 + config_dict["dataset"]["num_vocab"]

        self.cxt_embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.lbl_embedding = nn.Embedding(self.num_vocab-1, self.embed_dim)

    def forward(self, l_cxt, r_cxt, l_lbl, r_lbl):
        l_cxt_emb = self.compute_cxt_embed(l_cxt)
        r_cxt_emb = self.compute_cxt_embed(r_cxt)
        l_lbl_emb = self.lbl_embedding(torch.LongTensor(l_lbl)-self.num_vocab)
        r_lbl_emb = self.lbl_embedding(torch.LongTensor(r_lbl)-self.num_vocab)
        
        l_loss = torch.mul(l_cxt_emb, l_lbl_emb).squeeze()
        l_loss = torch.sum(l_loss, dim=1)
        l_loss = F.logsigmoid(-1 * l_loss)
        r_loss = torch.mul(r_cxt_emb, r_lbl_emb).squeeze()
        r_loss = torch.sum(r_loss, dim=1)
        r_loss = F.logsigmoid(r_loss)

        loss = torch.sum(l_loss) + torch.sum(r_loss)
        return -1 * loss

    def compute_cxt_embed(self, cxt):
        lbl_emb = self.cxt_embedding(torch.LongTensor(cxt))
        return torch.mean(lbl_emb, dim=1)
    
class Word2VecTrainer(nn.Module):
    def __init__(self, model, optimizer, config_dict):
        super(Word2VecTrainer, self).__init__()

        self.logger = logging.getLogger(__name__)

        self.model = model
        self.optim = optimizer
        self.config_dict = config_dict

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss, num_instances = 0, 0
        left_loader, right_loader = data_loader
        
        self.logger.info(f"-----------Epoch {epoch}/{self.config_dict['train']['epochs']}-----------")
        pbar = tqdm.tqdm(enumerate(left_loader), total=len(left_loader), desc="Training")

        right_iter = iter(right_loader)
        for batch_id, (l_cxt, l_lbl) in pbar:

            try:
                r_cxt, r_lbl = next(right_iter)
            except:
                right_iter = iter(right_loader)
                r_cxt, r_lbl = next(right_iter)

            l_cxt = l_cxt.to(dtype=torch.long)
            r_cxt = r_cxt.to(dtype=torch.long)
            l_lbl = l_lbl.to(dtype=torch.long)
            r_lbl = r_lbl.to(dtype=torch.long)

            loss = self.model(l_cxt, r_cxt, l_lbl, r_lbl)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss
            num_instances += r_lbl.size(0)

        train_loss = total_loss/num_instances

        return train_loss
        
    @torch.no_grad()
    def val_one_epoch(self, data_loader):
        self.model.eval()
        total_loss, num_instances = 0, 0
        left_loader, right_loader = data_loader

        pbar = tqdm.tqdm(enumerate(left_loader), total=len(left_loader), desc="Validation")

        right_iter = iter(right_loader)
        for batch_id, (l_cxt, l_lbl) in pbar:

            try:
                r_cxt, r_lbl = next(right_iter)
            except:
                right_iter = iter(right_loader)
                r_cxt, r_lbl = next(right_iter)

            l_cxt = l_cxt.to(dtype=torch.long)
            r_cxt = r_cxt.to(dtype=torch.long)
            l_lbl = l_lbl.to(dtype=torch.long)
            r_lbl = r_lbl.to(dtype=torch.long)

            loss = self.model(l_cxt, r_cxt, l_lbl, r_lbl)

            total_loss += loss
            num_instances += r_lbl.size(0)

        val_loss = total_loss/num_instances

        return val_loss

    def fit(self, train_loader, val_loader):
        logger = logging.getLogger(__name__)
        num_epochs = self.config_dict["train"]["epochs"]
        output_folder = self.config_dict["paths"]["output_folder"]

        best_val_loss = np.inf
        history = defaultdict(list)

        start = time.time()
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.val_one_epoch(val_loader)

            logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")
            
            history["train_loss"].append(float(train_loss.detach().numpy()))
            history["val_loss"].append(float(val_loss.detach().numpy()))

            if val_loss <= best_val_loss:
                logger.info(f"Validation Loss improved from {best_val_loss} to {val_loss}")
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(output_folder, "best_model.pt"))
            else:
                logger.info(f"Validation loss didn't improve from {best_val_loss}")

        end = time.time()
        time_taken = end-start
        logger.info('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60))
        logger.info(f"Best Val RMSE: {best_val_loss}")

        return history