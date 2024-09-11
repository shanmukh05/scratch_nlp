import os
import json
import torch
import logging
import pandas as pd

from preprocess.pos import PreprocessPOS
from .dataset import create_dataloader

class GRU:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        self.gru_ds = PreprocessPOS(self.config_dict)
        words, tags = self.gru_ds.get_data(self.gru_ds.corpus)

        train_loader, val_loader = create_dataloader(words, tags, self.val_split, self.batch_size, self.seed, "train")
    
    def run_infer(self):
        test_words, test_tags = self.gru_ds.get_data(self.gru_ds.test_corpus)

        test_loader = create_dataloader(test_words, None, None, self.batch_size, None, "train")

    def save_output(self):
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)