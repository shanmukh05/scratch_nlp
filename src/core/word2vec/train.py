import os
import tqdm
import time
import torch
import logging
import numpy as np
from collections import defaultdict

def train_one_epoch(model, optimizer, data_loader, epoch, config_dict):
    logger = logging.getLogger(__name__)
    model.train()
    total_loss, num_instances = 0, 0
    left_loader, right_loader = data_loader
    
    logger.info(f"-----------Epoch {epoch}/{config_dict['train']['epochs']}-----------")
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

        loss = model(l_cxt, r_cxt, l_lbl, r_lbl)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss
        num_instances += r_lbl.size(0)

    train_loss = total_loss/num_instances

    return train_loss
    
@torch.no_grad()
def val_one_epoch(model, data_loader, config_dict):
    model.eval()
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

        loss = model(l_cxt, r_cxt, l_lbl, r_lbl)

        total_loss += loss
        num_instances += r_lbl.size(0)

    val_loss = total_loss/num_instances

    return val_loss

def fit_model(model, optimizer, train_loader, val_loader, config_dict):
    logger = logging.getLogger(__name__)
    num_epochs = config_dict["train"]["epochs"]
    output_folder = config_dict["paths"]["output_folder"]

    best_val_loss = np.inf
    history = defaultdict(list)

    start = time.time()
    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model, optimizer, train_loader, epoch, config_dict)
        val_loss = val_one_epoch(model, val_loader, config_dict)

        logger.info(f"Train Loss : {train_loss} - Val Loss : {val_loss}")
        
        history["train_loss"].append(float(train_loss.detach().numpy()))
        history["val_loss"].append(float(val_loss.detach().numpy()))

        if val_loss <= best_val_loss:
            logger.info(f"Validation Loss improved from {best_val_loss} to {val_loss}")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, "best_model.pt"))
        else:
            logger.info(f"Validation loss didn't improve from {best_val_loss}")

    end = time.time()
    time_taken = end-start
    logger.info('Training completed in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_taken // 3600, (time_taken % 3600) // 60, (time_taken % 3600) % 60))
    logger.info(f"Best Val RMSE: {best_val_loss}")

    return history