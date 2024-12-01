import os
import json
import torch
import logging
import pandas as pd

from .dataset import create_dataloader, PreprocessGPT
from .model import GPTModel, GPTTrainer
from plot_utils import plot_embed, plot_history


class GPT:
    """
    A class to run GPT data preprocessing, training and inference

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
        Runs GPT Training and saves output
        """
        self.gpt_ds = PreprocessGPT(self.config_dict)
        tokens = self.gpt_ds.get_data()

        train_loader, val_loader = create_dataloader(
            tokens, "train", self.val_split, self.batch_size, self.seed
        )

        self.model = GPTModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = GPTTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()

    def run_infer(self):
        """
        Runs inference

        :return: True and Predicted tokens
        :rtype: tuple (list, list)
        """
        tokens = self.gpt_ds.get_test_data()

        test_loader = create_dataloader(
            tokens, "test", None, self.batch_size, self.seed
        )
        tokens, tokens_pred = self.trainer.generate(test_loader)

        tokens = self.gpt_ds.batched_ids2tokens(tokens)
        tokens_pred = self.gpt_ds.batched_ids2tokens(tokens_pred)

        return tokens, tokens_pred

    def save_output(self):
        """
        Saves Training and Inference results
        """
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), "w") as fp:
            json.dump(self.history, fp)

        src_embeds = self.model.embed_layer.weight.detach().numpy()
        vocab = list(self.gpt_ds.word2id.keys())
        plot_embed(src_embeds, vocab, output_folder, fname="Tokens Embeddings TSNE")

        plot_history(self.history, output_folder)

        tokens, tokens_pred = self.run_infer()
        test_df = pd.DataFrame.from_dict({"Sentence": tokens, "Generated": tokens_pred})
        test_df.to_csv(os.path.join(output_folder, "Test Predictions.csv"), index=False)
