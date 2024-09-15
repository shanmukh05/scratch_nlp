import os
import json
import torch
import logging
import pandas as pd

from preprocess.eng2tel import PreprocessSeq2Seq
from .dataset import create_dataloader
from .model import Seq2SeqModel, Seq2SeqTrainer
from plot_utils import plot_embed, plot_history


class Seq2Seq:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        self.seq2seq_ds = PreprocessSeq2Seq(self.config_dict)
        tokenSrc, tokenTgt = self.seq2seq_ds.get_data(self.seq2seq_ds.df)

        train_loader, val_loader = create_dataloader(tokenSrc, tokenTgt, self.val_split, self.batch_size, self.seed, "train")

        self.model = Seq2SeqModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Seq2SeqTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()
        return self.history

    def run_infer(self):
        tokenSrc, tokenTgt = self.seq2seq_ds.get_data(self.seq2seq_ds.test_df)
        test_loader = create_dataloader(tokenSrc, None, None, self.batch_size, None, "test")

        return test_loader
    
    def save_output(self):
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)

        src_embeds = self.model.encoder.src_embed_layer.weight.detach().numpy()
        src_vocab = list(self.seq2seq_ds.word2idSrc.keys())
        plot_embed(src_embeds, src_vocab, output_folder, fname="Source Embeddings TSNE")

        tgt_embeds = self.model.decoder.tgt_embed_layer.weight.detach().numpy()
        tgt_vocab = list(self.seq2seq_ds.word2idTgt.keys())
        plot_embed(tgt_embeds, tgt_vocab, output_folder, fname="Target Embeddings TSNE")

        plot_history(self.history, output_folder)
    

