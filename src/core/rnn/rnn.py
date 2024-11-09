import os
import json
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader

from preprocess.utils import preprocess_text
from .model import RNNModel, RNNTrainer
from .dataset import create_dataloader, RNNDataset
from plot_utils import plot_embed, plot_history, plot_conf_matrix

class RNN():
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """        
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

    def run(self):
        """
        _summary_
        """        
        self.rnn_ds = RNNDataset(self.config_dict)
        X, y = self.rnn_ds.get_data()
        self.config_dict["dataset"]["labels"] = list(self.rnn_ds.label_encoder.categories_[0])

        val_split = self.config_dict["dataset"]["val_split"]
        batch_size = self.config_dict["dataset"]["batch_size"]
        seed = self.config_dict["dataset"]["seed"]
        train_loader, val_loader = create_dataloader(X, y, val_split, batch_size, seed)

        self.model = RNNModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = RNNTrainer(self.model, optim, self.config_dict)

        self.history = self.trainer.fit(train_loader, val_loader)
        self.save_output()

    def run_infer(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        test_x, y_true = self.rnn_ds.get_test_data()
        test_ds = TensorDataset(torch.Tensor(test_x))
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.config_dict["dataset"]["batch_size"], 
            shuffle=True, drop_last=False, num_workers=1, pin_memory=True
            )
        
        y_pred = self.trainer.predict(test_loader)

        return y_true, y_pred


    def save_output(self):
        """
        _summary_
        """        
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        
        with open(os.path.join(output_folder, "training_history.json"), 'w') as fp:
            json.dump(self.history, fp)

        embeds = self.model.embed_layer.weight.detach().numpy()
        vocab = list(self.rnn_ds.word2id.keys())
        plot_embed(embeds, vocab, output_folder)

        plot_history(self.history, output_folder)

        y_true, y_pred = self.run_infer()
        y_true = self.rnn_ds.label_encoder.inverse_transform(y_true).squeeze()
        y_pred = self.rnn_ds.label_encoder.inverse_transform(y_pred).squeeze()
        classes = self.rnn_ds.label_encoder.categories_[0]
        plot_conf_matrix(y_true, y_pred, classes, output_folder)

