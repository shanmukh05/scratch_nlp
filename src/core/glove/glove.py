import os
import json
import torch
import logging

from preprocess.utils import preprocess_text
from .model import GloVeTrainer, GloVeModel
from .dataset import create_dataloader, GloVeDataset
from plot_utils import plot_embed
from plot_utils import plot_topk_cooccur_matrix


class GloVe:
    """
    A class to run GloVe data preprocessing, training and inference

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

    def run(self):
        """
        Runs GloVe Training and saves output
        """
        self.glove_ds = GloVeDataset(self.config_dict)
        X_ctr, X_cxt, X_cnt = self.glove_ds.get_data()

        val_split = self.config_dict["dataset"]["val_split"]
        batch_size = self.config_dict["dataset"]["batch_size"]
        seed = self.config_dict["dataset"]["seed"]
        train_loader, val_loader = create_dataloader(
            X_ctr, X_cxt, X_cnt, val_split, batch_size, seed
        )

        self.model = GloVeModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = GloVeTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()

    def get_embeddings(self, sentence):
        """
        Outputs Word embeddings

        :param sentence: Input sentence
        :type sentence: str
        :return: Word embeddings
        :rtype: torch.Tensor (seq_len, embed_dim)
        """
        operations = self.config_dict["preprocess"]["operations"]
        sentence = preprocess_text(sentence, operations)
        word_ls = sentence.split()
        word_ls = [i if i in self.glove_ds.word2id.keys() else "<UNK>" for i in word_ls]
        word_ids = [self.glove_ds.word2id[word] for word in word_ls]
        word_ids = torch.Tensor(word_ids).to(torch.long)
        word_embeds = self.model.ctr_embedding(word_ids)

        return word_embeds

    def save_output(self):
        """
        Saves Training and Inference results
        """
        output_folder = self.config_dict["paths"]["output_folder"]
        self.logger.info(f"Saving Outputs {output_folder}")

        with open(os.path.join(output_folder, "training_history.json"), "w") as fp:
            json.dump(self.history, fp)
        self.model.load_state_dict(
            torch.load(os.path.join(output_folder, "best_model.pt"), weights_only=True)
        )

        embeds = self.model.ctr_embedding.weight.detach().numpy()
        vocab = list(self.glove_ds.vocab_freq.keys())
        plot_embed(embeds, vocab, output_folder)

        plot_topk_cooccur_matrix(self.glove_ds.cooccur_mat, vocab, output_folder)
