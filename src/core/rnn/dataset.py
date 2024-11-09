import torch
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from preprocess.imdb_reviews import PreprocessIMDB
from core.word2vec.dataset import Word2VecDataset


class RNNDataset(Word2VecDataset):
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """        
        self.logger = logging.getLogger(__name__)

        self.config_dict = config_dict
        self.num_vocab = config_dict["dataset"]["num_vocab"]
        self.preprocess()
        self.get_vocab()

    def get_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        self.word2id["<PAD>"] = len(self.word2id)
        self.id2word[len(self.id2word)] = "<PAD>"

        self.label_encoder = OneHotEncoder()

        X = self.pad_slice_text(self.text_ls)

        y = np.array(self.preproc_cls.label_ls).reshape(-1, 1)
        y = self.label_encoder.fit_transform(y).toarray()
        return np.array(X), y
    
    def get_test_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        root_path = self.config_dict["paths"]["test_folder"]
        explore_folder = self.config_dict["dataset"]["explore_folder"]
        num_samples = self.config_dict["dataset"]["test_samples"]
        operations = self.config_dict["preprocess"]["operations"]
        randomize = self.config_dict["preprocess"]["randomize"]

        test_preproc_cls = PreprocessIMDB(
            root_path, explore_folder, num_samples, operations, randomize
        )
        test_preproc_cls.run()
        test_text_ls, test_labels = test_preproc_cls.text_ls, test_preproc_cls.label_ls

        X_test = self.pad_slice_text(test_text_ls)
        y_test = np.array(test_labels).reshape(-1, 1)
        y_test = self.label_encoder.transform(y_test).toarray()

        return X_test, y_test

    def pad_slice_text(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """        
        seq_len = self.config_dict["dataset"]["seq_len"]
        X = []
        for text in text_ls:
            ls = []
            for word in text.split()[:seq_len]:
                if word in self.word2id.keys():
                    ls.append(self.word2id[word])
                else:
                    ls.append(self.word2id["<UNK>"])
            if len(ls) < seq_len:
                num_pad = seq_len - len(ls)
                ls.extend([self.word2id["<PAD>"]]*num_pad)
            X.append(ls)

        return X
    
def create_dataloader(X, y, val_split, batch_size, seed):
    """
    _summary_

    :param X: _description_
    :type X: _type_
    :param y: _description_
    :type y: _type_
    :param val_split: _description_
    :type val_split: _type_
    :param batch_size: _description_
    :type batch_size: _type_
    :param seed: _description_
    :type seed: _type_
    :return: _description_
    :rtype: _type_
    """    
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=val_split, random_state=seed)
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=val_split, random_state=seed)

    train_ds = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

    val_ds = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader