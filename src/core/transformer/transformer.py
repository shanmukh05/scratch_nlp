import os
import json
import torch
import logging
import pandas as pd

from .dataset import create_dataloader, PreprocessTransformer
from .model import TransformerModel, TransformerTrainer
from plot_utils import plot_embed, plot_history


class Transformer:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.test_split = self.config_dict["dataset"]["test_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        self.transformer_ds = PreprocessTransformer(self.config_dict)
        tokens = self.transformer_ds.get_data()

        train_loader, val_loader, self.test_loader = create_dataloader(tokens, self.val_split, self.test_split, self.batch_size, self.seed)

        self.model = TransformerModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = TransformerTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()
    
    def run_infer(self):
        sents, tokens_pred = self.trainer.predict(self.test_loader)
        sents = self.transformer_ds.batched_ids2tokens(sents)
        tokens_pred = tokens_pred.argmax(axis=-1).astype("int")
        tokens_pred = self.transformer_ds.batched_ids2tokens(tokens_pred)
        
        return sents, tokens_pred
    
    def save_output(self):
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)

        src_embeds = self.model.src_embed_layer.weight.detach().numpy()
        vocab = list(self.transformer_ds.word2id.keys())
        plot_embed(src_embeds, vocab, output_folder, fname="Tokens Embeddings TSNE")

        plot_history(self.history, output_folder)

        sents, tokens_pred = self.run_infer()
        test_df = pd.DataFrame.from_dict({
            "Sentence": sents,
            "Prediction": tokens_pred
        })
        test_df.to_csv(os.path.join(output_folder, "Test Predictions.csv"), index=False)