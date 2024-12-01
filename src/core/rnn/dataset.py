import torch
import logging
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from preprocess.imdb_reviews import PreprocessIMDB
from core.word2vec.dataset import Word2VecDataset


class RNNDataset(Word2VecDataset):
    """
    RNN Dataset

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.config_dict = config_dict
        self.num_vocab = config_dict["dataset"]["num_vocab"]
        self.preprocess()
        self.get_vocab()

    def get_data(self):
        """
        Generates tokens and labels from extracted data

        :return: Input tokens, Labels
        :rtype: tuple (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, num_classes])
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
        Generates test tokens and labels from extracted data

        :return: Input tokens, Labels
        :rtype: tuple (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, num_classes])
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
        Pads and slices text to seq_len tokens

        :param text_ls: List of text samples
        :type text_ls: list
        :return: list of preprocessed text
        :rtype: list
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
                ls.extend([self.word2id["<PAD>"]] * num_pad)
            X.append(ls)

        return X


def create_dataloader(X, y, val_split=0.2, batch_size=32, seed=2024):
    """
    Creates Train, Validation DataLoader

    :param X: Input tokens
    :type X: torch.Tensor (num_samples, seq_len)
    :param y: Output Labels
    :type y: torch.Tensor (num_samples, num_classes)
    :param val_split: validation split, defaults to 0.2
    :type val_split: float, optional
    :param batch_size: Batch size, defaults to 32
    :type batch_size: int, optional
    :param seed: Seed, defaults to 2024
    :type seed: int, optional
    :return: Train, Val dataloaders
    :rtype: tuple (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )

    train_ds = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    val_ds = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader
