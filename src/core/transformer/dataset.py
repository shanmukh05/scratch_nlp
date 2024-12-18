import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from preprocess.utils import preprocess_text, BytePairEncoding


class PreprocessTransformer:
    """
    A class to preprocess Transformer Data

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.input_path = config_dict["paths"]["input_file"]
        self.num_samples = config_dict["dataset"]["num_samples"]
        self.seq_len = 1 + config_dict["dataset"]["seq_len"]
        self.operations = config_dict["preprocess"]["operations"]

        self.bpe = BytePairEncoding(config_dict)

    def get_data(self):
        """
        Converts extracted data into tokens

        :return: Text tokens
        :rtype: numpy.ndarray (num_samples, seq_len)
        """
        text_ls = self.extract_data()
        text_ls = self.preprocess_text(text_ls)

        bpe_words = self.get_vocab(text_ls)
        text_tokens = np.zeros((len(text_ls), self.seq_len))

        text_lens = [len(text.split()) for text in text_ls]
        count = 0
        for i, text_len in enumerate(text_lens):
            text = " ".join(bpe_words[count : count + text_len]).split()
            count += text_len
            text = text[: self.seq_len]

            if len(text) < self.seq_len:
                text = text + ["<PAD>"] * (self.seq_len - len(text))

            text_tokens[i] = np.array([self.word2id[ch] for ch in text])

        return text_tokens

    def preprocess_text(self, text_ls):
        """
        Preprocesses list of strings

        :param text_ls: List of Raw strings
        :type text_ls: list
        :return: List of preprocssed strings
        :rtype: list
        """
        text_ls = [preprocess_text(text, self.operations) for text in text_ls]

        return text_ls

    def get_vocab(self, text_ls):
        """
        Generates Vocabulary

        :param text_ls: List of preprocessed strings
        :type text_ls: list
        :return: Corpus generated by WordPiece
        :rtype: list
        """
        self.logger.info("Building Vocabulary using Byte Pair Encoding method")
        words = self.bpe.fit(text_ls)
        vocab = ["<PAD>"] + list(self.bpe.vocab_freq.keys())

        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {v: k for k, v in self.word2id.items()}

        return words

    def batched_ids2tokens(self, tokens):
        """
        Converting sentence of ids to tokens

        :param tokens: Tokens Array, 2D array (num_samples, seq_len)
        :type tokens: numpy.ndarray
        :return: List of decoded sentences
        :rtype: list
        """
        func = lambda x: self.id2word[x]
        vect_func = np.vectorize(func)

        tokens = vect_func(tokens)

        sentences = []
        for words in tokens:
            txt = ""
            for word in words:
                if word not in ["<PAD>"]:
                    txt += f"{word} "
            txt = txt[:-1]
            txt = txt.split("</w>")
            txt = " ".join([i.replace(" ", "") for i in txt])
            sentences.append(txt)

        return sentences

    def extract_data(self):
        """
        Extracts data from Lyrics csv file

        :return: Lost of raw strings
        :rtype: list
        """
        df = pd.read_csv(self.input_path, nrows=self.num_samples)

        return df["lyrics"]


def create_dataloader(X, val_split=0.2, test_split=0.2, batch_size=32, seed=2024):
    """
    Creates Train, Validation and Test DataLoader

    :param X: Input tokens
    :type X: torch.Tensor (num_samples, seq_len+1)
    :param val_split: validation split, defaults to 0.2
    :type val_split: float, optional
    :param test_split: test split, defaults to 0.2
    :type test_split: float, optional
    :param batch_size: Batch size, defaults to 32
    :type batch_size: int, optional
    :param seed: Seed, defaults to 2024
    :type seed: int, optional
    :return: Train, Val, Test dataloaders
    :rtype: tuple (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader)
    """
    train_X, val_X = train_test_split(
        X, test_size=val_split + test_split, random_state=seed
    )
    val_X, test_X = train_test_split(
        X, test_size=test_split / (val_split + test_split), random_state=seed
    )

    train_ds = TensorDataset(torch.Tensor(train_X))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    val_ds = TensorDataset(torch.Tensor(val_X))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    test_ds = TensorDataset(torch.Tensor(test_X))
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
