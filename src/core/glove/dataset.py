import torch
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from core.word2vec.dataset import Word2VecDataset


class GloVeDataset(Word2VecDataset):
    """
    GloVe Dataset

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict
        self.num_vocab = config_dict["dataset"]["num_vocab"]
        self.context = config_dict["dataset"]["context"]

        self.id2word = {}
        self.word2id = {}

        self.preprocess()
        self.get_vocab()

    def get_data(self):
        """
        Generates Coccurence Matrix

        :return: Center, Context words and Co-occurence matrix
        :rtype: tuple (numpy.ndarray [num_samples, ], numpy.ndarray [num_samples, ], numpy.ndarray [num_samples, ])
        """
        self.cooccur_mat = np.zeros((1 + self.num_vocab, 1 + self.num_vocab))

        for text in self.text_ls:
            words = text.split()

            if len(words) < 1 + self.context:
                continue

            for i in range(self.context, len(words) - self.context):
                id_i = self.word2id[words[i]]
                for j in range(i - self.context, i + self.context):
                    id_j = self.word2id[words[j]]
                    dist = np.abs(j - i)
                    if dist != 0:
                        self.cooccur_mat[id_i][id_j] += 1 / dist

        X_ctr, X_cxt = np.indices((1 + self.num_vocab, 1 + self.num_vocab))
        X_ctr, X_cxt = X_ctr.flatten(), X_cxt.flatten()
        X_cnt = self.cooccur_mat.flatten()

        return X_ctr, X_cxt, X_cnt


def create_dataloader(X_ctr, X_cxt, X_count, val_split=0.2, batch_size=32, seed=2024):
    """
    Creates Train, Validation DataLoader

    :param X_ctr: Center words
    :type X_ctr: numpy.ndarray (num_samples, )
    :param X_cxt: Context words
    :type X_cxt: numpy.ndarray (num_samples, )
    :param X_count: Co-occurence matrix elements
    :type X_count: numpy.ndarray (num_samples, )
    :param val_split: validation split, defaults to 0.2
    :type val_split: float, optional
    :param batch_size: Batch size, defaults to 32
    :type batch_size: int, optional
    :param seed: Seed, defaults to 2024
    :type seed: int, optional
    :return: Train, Val dataloaders
    :rtype: tuple (torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    train_ctr, val_ctr, train_cxt, val_cxt, train_count, val_count = train_test_split(
        X_ctr, X_cxt, X_count, test_size=val_split, random_state=seed
    )

    train_ds = TensorDataset(
        torch.Tensor(train_ctr), torch.Tensor(train_cxt), torch.Tensor(train_count)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    val_ds = TensorDataset(
        torch.Tensor(val_ctr), torch.Tensor(val_cxt), torch.Tensor(val_count)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    return train_loader, val_loader
