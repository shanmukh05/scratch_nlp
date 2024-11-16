import torch
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from core.word2vec.dataset import Word2VecDataset


class GloVeDataset(Word2VecDataset):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
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
        _summary_

        :return: _description_
        :rtype: _type_
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


def create_dataloader(X_ctr, X_cxt, X_count, val_split, batch_size, seed):
    """
    _summary_

    :param X_ctr: _description_
    :type X_ctr: _type_
    :param X_cxt: _description_
    :type X_cxt: _type_
    :param X_count: _description_
    :type X_count: _type_
    :param val_split: _description_
    :type val_split: _type_
    :param batch_size: _description_
    :type batch_size: _type_
    :param seed: _description_
    :type seed: _type_
    :return: _description_
    :rtype: _type_
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
