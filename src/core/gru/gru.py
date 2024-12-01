import os
import json
import torch
import logging
import pandas as pd

from preprocess.pos import PreprocessPOS
from .dataset import create_dataloader
from .model import GRUModel, GRUTrainer
from plot_utils import plot_embed, plot_history, plot_conf_matrix


class GRU:
    """
    A class to run GRU data preprocessing, training and inference

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        """
        Runs GRU Training and saves output
        """
        self.gru_ds = PreprocessPOS(self.config_dict)
        words, tags = self.gru_ds.get_data(self.gru_ds.corpus)
        self.config_dict["dataset"]["num_classes"] = len(self.gru_ds.unq_pos)
        self.config_dict["dataset"]["labels"] = self.gru_ds.unq_pos

        train_loader, val_loader = create_dataloader(
            words, tags, self.val_split, self.batch_size, self.seed, "train"
        )

        self.model = GRUModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = GRUTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()

    def run_infer(self):
        """
        Runs inference

        :return: True and Predicted labels
        :rtype: tuple (torch.Tensor [num_samples, seq_len], torch.Tensor [num_samples, seq_len])
        """
        test_words, test_tags = self.gru_ds.get_data(self.gru_ds.test_corpus)
        test_tags = test_tags.argmax(-1).flatten()

        test_loader = create_dataloader(
            test_words, None, None, self.batch_size, None, "test"
        )
        test_tags_pred = self.trainer.predict(test_loader)

        return test_tags, test_tags_pred

    def save_output(self):
        """
        Saves Training and Inference results
        """
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), "w") as fp:
            json.dump(self.history, fp)

        embeds = self.model.embed_layer.weight.detach().numpy()
        vocab = list(self.gru_ds.word2idX.keys())
        plot_embed(embeds, vocab, output_folder)

        plot_history(self.history, output_folder)

        y_true, y_pred = self.run_infer()
        y_true = [self.gru_ds.posDec[i] for i in y_true]
        y_pred = [self.gru_ds.posDec[i] for i in y_pred]
        classes = self.gru_ds.unq_pos
        plot_conf_matrix(y_true, y_pred, classes, output_folder)
