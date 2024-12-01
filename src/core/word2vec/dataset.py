import os
import torch
import logging
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from preprocess.imdb_reviews import PreprocessIMDB
from .huffman import HuffmanBTree


class Word2VecDataset:
    """
    Word2Vec Dataset

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict
        self.num_vocab = config_dict["dataset"]["num_vocab"]

        self.id2word = {}
        self.word2id = {}

        self.preprocess()
        self.get_vocab()
        self.huffman = HuffmanBTree(self.vocabidx_freq)

    def preprocess(self):
        """
        Preprocessed extracted data
        """
        root_path = self.config_dict["paths"]["input_folder"]
        explore_folder = self.config_dict["dataset"]["explore_folder"]
        num_samples = self.config_dict["dataset"]["num_samples"]
        operations = self.config_dict["preprocess"]["operations"]
        randomize = self.config_dict["preprocess"]["randomize"]

        self.preproc_cls = PreprocessIMDB(
            root_path, explore_folder, num_samples, operations, randomize
        )
        self.logger.info("Text Preprocessing Done")
        self.preproc_cls.run()
        self.text_ls = self.preproc_cls.text_ls

    def get_vocab(self):
        """
        Generates vocabulary from from preprocessed text
        """
        self.vocab_freq = {}

        all_text = " ".join(self.text_ls)
        for vocab in all_text.split():
            self.vocab_freq[vocab] = self.vocab_freq.get(vocab, 0) + 1

        self.vocab_freq = dict(
            sorted(self.vocab_freq.items(), key=lambda k: k[1], reverse=True)
        )
        self.vocab_freq = dict(
            itertools.islice(self.vocab_freq.items(), self.num_vocab)
        )

        self.vocab_freq["<UNK>"] = 0
        for i, text in enumerate(self.text_ls):
            new_text = []
            for word in text.split():
                if word not in self.vocab_freq.keys():
                    self.vocab_freq["<UNK>"] += 1
                    new_text.append("<UNK>")
                else:
                    new_text.append(word)
            self.text_ls[i] = " ".join(new_text)

        self.word2id = {w: i for i, w in enumerate(self.vocab_freq.keys())}
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.vocabidx_freq = {self.word2id[k]: v for k, v in self.vocab_freq.items()}
        self.logger.info("Vocabulary Bulding done")

    def make_pairs(self):
        """
        Creates Left and Right context and Labels using Huffman binary tree

        :return: Left, Right Context, Left, Right Label
        :rtype: tuple (list, list, list, list)
        """
        self.context = self.config_dict["dataset"]["context"]
        left_cxt_huff, right_cxt_huff, left_lbl_huff, right_lbl_huff = [], [], [], []

        for text in self.text_ls:
            words = text.split()

            if len(words) < 1 + self.context:
                continue

            for i in range(self.context, len(words) - self.context):
                l_idx = [self.word2id[words[i - j]] for j in range(1, self.context)]
                r_idx = [self.word2id[words[i + j]] for j in range(1, self.context)]
                cxt_idx = l_idx + r_idx
                lbl_idx = self.word2id[words[i]]

                left_huff_lbl_idx = self.huffman.left_huff_dict[lbl_idx]
                right_huff_lbl_idx = self.huffman.right_huff_dict[lbl_idx]

                left_cxt_huff.extend([cxt_idx] * len(left_huff_lbl_idx))
                right_cxt_huff.extend([cxt_idx] * len(right_huff_lbl_idx))
                left_lbl_huff.extend(left_huff_lbl_idx)
                right_lbl_huff.extend(right_huff_lbl_idx)
        return left_cxt_huff, right_cxt_huff, left_lbl_huff, right_lbl_huff


def create_dataloader(
    left_cxt, right_cxt, left_lbl, right_lbl, val_split=0.2, batch_size=32, seed=2024
):
    """
    Creates Train, Validation left and Right DataLoader

    :param left_cxt: Left context
    :type left_cxt: list
    :param right_cxt: Right context
    :type right_cxt: list
    :param left_lbl: Left label
    :type left_lbl: list
    :param right_lbl: Right label
    :type right_lbl: list
    :param val_split: validation split, defaults to 0.2
    :type val_split: float
    :param batch_size: Batch size, defaults to 32
    :type batch_size: int
    :param seed: Seed, defaults to 2024
    :type seed: int, optional
    :return: train, val left and right dataloader
    :rtype: tuple (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    logger = logging.getLogger(__name__)
    train_left_cxt, val_left_cxt, train_left_lbl, val_left_lbl = train_test_split(
        left_cxt, left_lbl, test_size=val_split, random_state=seed
    )
    train_right_cxt, val_right_cxt, train_right_lbl, val_right_lbl = train_test_split(
        right_cxt, right_lbl, test_size=val_split, random_state=seed
    )
    logger.info("Splitted data")

    train_left_ds = TensorDataset(
        torch.Tensor(train_left_cxt), torch.Tensor(train_left_lbl)
    )
    train_left_loader = DataLoader(
        train_left_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    train_right_ds = TensorDataset(
        torch.Tensor(train_right_cxt), torch.Tensor(train_right_lbl)
    )
    train_right_loader = DataLoader(
        train_right_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    val_left_ds = TensorDataset(torch.Tensor(val_left_cxt), torch.Tensor(val_left_lbl))
    val_left_loader = DataLoader(
        val_left_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    val_right_ds = TensorDataset(
        torch.Tensor(val_right_cxt), torch.Tensor(val_right_lbl)
    )
    val_right_loader = DataLoader(
        val_right_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )
    logger.info(
        "Created Training and Validation Data Loaders for Left and Right Branches"
    )
    return train_left_loader, train_right_loader, val_left_loader, val_right_loader
