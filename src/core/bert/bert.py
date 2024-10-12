import os
import json
import torch
import logging
import pandas as pd

from .dataset_pretrain import create_dataloader_pretrain, PreprocessBERTPretrain
from .model import BERTModel
from .pretrain import BERTPretrainTrainer
from plot_utils import plot_embed, plot_history

class BERT:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.test_split = self.config_dict["dataset"]["test_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        self.trainer_pretrain, self.pretrain_history = self.run_pretrain()
        self.model_pretrain = self.trainer_pretrain.model

        self.save_output()

    def run_pretrain(self):
        self.bert_pretrain_ds = PreprocessBERTPretrain(self.config_dict)
        text_tokens, nsp_labels = self.bert_pretrain_ds.get_data()
        word2id = self.bert_pretrain_ds.word2id

        train_loader, val_loader, self.test_loader_pretrain = create_dataloader_pretrain(
            text_tokens, nsp_labels, 
            self.config_dict, word2id, self.val_split, 
            self.test_split, self.batch_size, self.seed
            )

        model = BERTModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = BERTPretrainTrainer(model, optim, self.config_dict)

        history = trainer.fit(train_loader, val_loader)

        return trainer, history


    def save_output(self):
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "pretraining_history.json"), 'w') as fp:
            json.dump(self.pretrain_history, fp)

        embeds = self.model_pretrain.embed_layer.weight.detach().numpy()
        vocab = list(self.bert_pretrain_ds.word2id.keys())
        plot_embed(embeds, vocab, output_folder, fname="Tokens Embeddings TSNE")

        plot_history(self.pretrain_history, output_folder, "Pretrain History")

    