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
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """
        self.seq2seq_ds = PreprocessSeq2Seq(self.config_dict)
        tokenSrc, tokenTgt = self.seq2seq_ds.get_data(self.seq2seq_ds.df)

        train_loader, val_loader = create_dataloader(
            tokenSrc, tokenTgt, self.val_split, self.batch_size, self.seed, "train"
        )

        self.model = Seq2SeqModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Seq2SeqTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()
        return self.history

    def run_infer(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """
        tokens_src, tokens_tgt = self.seq2seq_ds.get_data(self.seq2seq_ds.test_df)
        test_loader = create_dataloader(
            tokens_src, None, None, self.batch_size, None, "test"
        )

        tokens_tgt_pred = self.trainer.predict(test_loader)
        tokens_tgt_pred = tokens_tgt_pred.argmax(axis=-1).astype("int")

        sents_src = self.seq2seq_ds.batched_ids2tokens(tokens_src, "src")
        sents_tgt = self.seq2seq_ds.batched_ids2tokens(tokens_tgt, "tgt")
        sents_tgt_pred = self.seq2seq_ds.batched_ids2tokens(tokens_tgt, "tgt")

        return sents_src, sents_tgt, sents_tgt_pred

    def save_output(self):
        """
        _summary_
        """
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), "w") as fp:
            json.dump(self.history, fp)

        src_embeds = self.model.encoder.src_embed_layer.weight.detach().numpy()
        src_vocab = list(self.seq2seq_ds.word2idSrc.keys())
        plot_embed(src_embeds, src_vocab, output_folder, fname="Source Embeddings TSNE")

        tgt_embeds = self.model.decoder.tgt_embed_layer.weight.detach().numpy()
        tgt_vocab = list(self.seq2seq_ds.word2idTgt.keys())
        plot_embed(tgt_embeds, tgt_vocab, output_folder, fname="Target Embeddings TSNE")

        plot_history(self.history, output_folder)

        sents_src, sents_tgt, sents_tgt_pred = self.run_infer()
        test_df = pd.DataFrame.from_dict(
            {"Source": sents_src, "Target": sents_tgt, "Prediction": sents_tgt_pred}
        )
        test_df.to_csv(os.path.join(output_folder, "Test Predictions.csv"), index=False)
