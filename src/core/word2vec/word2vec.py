import os
import json
import torch
import logging

from preprocess.utils import preprocess_text
from .model import Word2VecModel, Word2VecTrainer
from .dataset import create_dataloader, Word2VecDataset
from .utils import plot_embed

class Word2Vec():
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

    def run(self):
        self.cbow_ds = Word2VecDataset(self.config_dict)
        l_cxt, r_cxt, l_lbl, r_lbl = self.cbow_ds.make_pairs()

        val_split = self.config_dict["dataset"]["val_split"]
        batch_size = self.config_dict["dataset"]["batch_size"]
        seed = self.config_dict["dataset"]["seed"]
        train_left_loader, train_right_loader, val_left_loader, val_right_loader = create_dataloader(l_cxt, r_cxt, l_lbl, r_lbl, val_split, batch_size, seed)
        train_loader = (train_left_loader, train_right_loader)
        val_loader = (val_left_loader, val_right_loader)
        
        self.model = Word2VecModel(self.config_dict)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config_dict["train"]["lr"])
        self.trainer = Word2VecTrainer(self.model, optim, self.config_dict)
        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()

    def get_embeddings(self, sentence):
        operations = self.config_dict["preprocess"]["operations"]
        sentence = preprocess_text(sentence, operations)
        word_ls = sentence.split()
        word_ls = [i if i in self.cbow_ds.word2id.keys() else "<UNK>" for i in word_ls]
        word_ids = [self.cbow_ds.word2id[word] for word in word_ls]
        word_ids = torch.Tensor(word_ids).to(torch.long)
        word_embeds = self.model.cxt_embedding(word_ids)

        return word_embeds

    def save_output(self):
        output_folder = self.config_dict["paths"]["output_folder"]
        self.logger.info(f"Saving Outputs {output_folder}")
        
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)
        self.model.load_state_dict(torch.load(os.path.join(output_folder, "best_model.pt"), weights_only=True))
        embeds = self.model.cxt_embedding.weight.detach().numpy()
        vocab = list(self.cbow_ds.vocab_freq.keys())
        plot_embed(embeds, vocab, output_folder)

