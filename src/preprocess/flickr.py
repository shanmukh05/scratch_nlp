import os
import logging
import numpy as np
import pandas as pd
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import preprocess_text


class PreprocessFlickr:
    """
    Preprocessing Flickr Dataset

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.config_dict = config_dict
        self.train_df, self.test_df = self.extract_data()

    def get_data(self):
        """
        Preprocessing

        :return: Returns image paths, Train Tokens, (Train, Test Transforms)
        :rtype: (list, numpy.ndarray [num_samples, seq_len], (albumentations.Compose, albumentations.Compose))
        """
        self.get_vocab(self.train_df)
        train_tokens = self.word_tokens(self.train_df)

        train_transforms, test_transforms = self.image_transforms("train")

        return (
            list(self.train_df["Path"]),
            train_tokens,
            (train_transforms, test_transforms),
        )

    def get_test_data(self):
        """
        Generating test Data

        :return: Returns image paths, Test Tokens, Test Transforms
        :rtype: (list, numpy.ndarray [num_samples, seq_len], albumentations.Compose)
        """
        test_tokens = self.word_tokens(self.test_df)
        test_transforms = self.image_transforms("test")

        return list(self.test_df["Path"]), test_tokens, test_transforms

    def extract_data(self):
        """
        Extracting Image and Captions Data

        :return: Train and Test DataFrames
        :rtype: tuple (pandas.DataFrame, pandas.DataFrame)
        """
        self.logger.info(
            "Creating a DataFrame from images folder and captions txt file"
        )
        im_folder = self.config_dict["paths"]["image_folder"]
        caption_file = self.config_dict["paths"]["captions_file"]
        operations = self.config_dict["preprocess"]["operations"]

        with open(caption_file, "r") as f:
            lines = np.array(f.readlines()[1:])

        num_train = self.config_dict["dataset"]["train_samples"]
        num_test = self.config_dict["dataset"]["test_samples"]
        rand_ids = np.random.choice(len(lines), num_train + num_test)

        paths, captions = zip(*(s.split(",") for s in lines[rand_ids]))
        paths = [os.path.join(im_folder, i) for i in paths]
        df = pd.DataFrame.from_dict({"Path": paths, "Caption": captions})
        df["Caption"] = df["Caption"].map(lambda x: preprocess_text(x, operations))

        train_df = df.iloc[:num_train]
        test_df = df.iloc[num_train:]

        return train_df, test_df

    def get_vocab(self, train_df):
        """
        Generates Vocabulary

        :param train_df: DataFrame with Training Captions
        :type train_df: pandas.DataFrame
        """
        self.logger.info("Building Vocabulary from training data captions")
        num_vocab = (
            self.config_dict["dataset"]["num_vocab"]
            - self.config_dict["dataset"]["num_extra_tokens"]
        )
        all_words = []

        for text in train_df["Caption"]:
            all_words += text.split()

        topk_vocab_freq = Counter(all_words).most_common(num_vocab)
        self.vocab = ["<START>", "<END>", "<PAD>", "<UNK>"] + [
            i[0] for i in topk_vocab_freq
        ]
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {v: k for k, v in self.word2id.items()}

    def word_tokens(self, df):
        """
        Coverting Sentences to Tokens

        :param df: Captions DataFrame
        :type df: pandas.DataFrame
        :return: Tokens array (num_samples, seq_len)
        :rtype: numpy.ndarray
        """
        seq_len = self.config_dict["dataset"]["seq_len"]

        tokens = np.zeros((len(df), seq_len))

        for i, text in enumerate(df["Caption"]):
            words = ["<START>"] + text.split()[: seq_len - 2]
            if len(words) < seq_len - 1:
                words += ["<PAD>"] * (seq_len - 1 - len(words))
            words += ["<END>"]

            for j, w in enumerate(words):
                if w in self.vocab:
                    tokens[i, j] = self.word2id[w]
                else:
                    tokens[i, j] = self.word2id["<UNK>"]

        return tokens

    def batched_ids2captions(self, tokens):
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

        captions = []
        for words in tokens:
            txt = ""
            for word in words:
                if word not in ["<START>", "<END>", "<PAD>"]:
                    txt += f"{word} "
            captions.append(txt[:-1])
        return captions

    def image_transforms(self, data_type):
        """
        Creating Albumentations Transforms for train or test data

        :param data_type: {'train', 'test'}. Type of Data
        :type data_type: str
        :return: Transforms
        :rtype: albumentations.Compose
        """
        im_w, im_h = self.config_dict["preprocess"]["image_dim"][1:]
        train_transforms = A.Compose(
            [
                A.Resize(im_w, im_h),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        )

        test_transforms = A.Compose(
            [
                A.Resize(im_w, im_h),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        )

        if data_type == "train":
            return train_transforms, test_transforms
        else:
            return test_transforms
