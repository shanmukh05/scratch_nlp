import os
import json
import torch
import logging
import pandas as pd

from preprocess.flickr import PreprocessFlickr
from .model import LSTMModel, LSTMTrainer
from .dataset import create_dataloader
from plot_utils import plot_history, plot_embed


class LSTM:
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """        
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

        self.val_split = self.config_dict["dataset"]["val_split"]
        self.batch_size = self.config_dict["dataset"]["batch_size"]
        self.seed = self.config_dict["dataset"]["seed"]

    def run(self):
        """
        _summary_
        """        
        self.lstm_ds = PreprocessFlickr(self.config_dict)
        train_paths, train_tokens, transforms = self.lstm_ds.get_data()

        train_loader, val_loader = create_dataloader(train_paths, train_tokens, transforms, self.val_split, self.batch_size, self.seed, "train")

        self.model = LSTMModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = LSTMTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()
    
    
    def run_infer(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        test_paths, test_tokens, transforms = self.lstm_ds.get_test_data()
        test_loader = create_dataloader(test_paths, test_tokens, transforms, self.val_split, self.batch_size, self.seed, "test")

        test_pred_tokens = self.trainer.predict(test_loader)
        test_pred_tokens = test_pred_tokens.argmax(axis=-1).astype("int")

        test_captions = self.lstm_ds.batched_ids2captions(test_tokens)
        test_pred_captions = self.lstm_ds.batched_ids2captions(test_pred_tokens)

        return test_paths, test_captions, test_pred_captions
    
    def save_output(self):
        """
        _summary_
        """        
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)

        embeds = self.model.embed_layer.weight.detach().numpy()
        vocab = list(self.lstm_ds.word2id.keys())
        plot_embed(embeds, vocab, output_folder)

        plot_history(self.history, output_folder)

        test_paths, test_captions, test_pred_captions = self.run_infer()
        test_df = pd.DataFrame.from_dict({
            "Image": [os.path.basename(i) for i in test_paths],
            "True Caption": test_captions,
            "Generated Caption": test_pred_captions
        })
        test_df.to_csv(os.path.join(output_folder, "Test Predictions.csv"), index=False)