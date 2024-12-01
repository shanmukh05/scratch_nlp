import re
import logging
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter


class PreprocessSeq2Seq:
    """
    Loading Data and Generating Source, target Data for SEQ2SEQ Model Training.

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """

    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.config_dict = config_dict
        self.num_src_vocab = config_dict["dataset"]["num_src_vocab"]
        self.num_tgt_vocab = config_dict["dataset"]["num_tgt_vocab"]
        self.seq_len = config_dict["dataset"]["seq_len"]

        self.df, self.test_df = self.extract_data()
        self.get_vocab(self.df)

    def get_data(self, df):
        """
        Generating Source and Target Array of Tokens

        :param df: DataFrame with Source and Target sentences
        :type df: pandas.DataFrame
        :return: Source and Target Arrays
        :rtype: tuple (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, seq_len])
        """
        src = list(df["Source"].map(lambda x: self.preprocess_src(x)))
        tgt = list(df["Target"].map(lambda x: self.preprocess_tgt(x)))

        tokenSrc = np.zeros((len(src), self.seq_len))
        tokenTgt = np.zeros((len(tgt), self.seq_len))

        for i, (s, t) in enumerate(zip(src, tgt)):
            s = ["<SOS>"] + s[: self.seq_len - 2] + ["<EOS>"]
            t = ["<SOS>"] + t[: self.seq_len - 2] + ["<EOS>"]

            if len(s) < self.seq_len:
                s = s + ["<PAD>"] * (self.seq_len - len(s))
            if len(t) < self.seq_len:
                t = s + ["<PAD>"] * (self.seq_len - len(t))

            for j, (s_w, t_w) in enumerate(zip(s, t)):
                bool_src = s_w in self.vocabSrc
                tokenSrc[i][j] = (
                    self.word2idSrc[s_w] if bool_src else self.word2idSrc["<UNK>"]
                )

                bool_tgt = t_w in self.vocabTgt
                tokenTgt[i][j] = (
                    self.word2idTgt[t_w] if bool_tgt else self.word2idTgt["<UNK>"]
                )

        return tokenSrc, tokenTgt

    def get_vocab(self, df):
        """
        Generates Vocabulary

        :param df: DataFrame with Source and Target sentences
        :type df: pandas.DataFrame
        """
        self.logger.info(
            "Building Vocabulary for Source and Target Languages using Training data"
        )

        src = list(df["Source"].map(lambda x: self.preprocess_src(x)))
        tgt = list(df["Target"].map(lambda x: self.preprocess_tgt(x)))

        all_src_words = [word for sent in src for word in sent]
        topk_src_vocab_freq = Counter(all_src_words).most_common(self.num_src_vocab - 4)
        self.vocabSrc = np.array(
            ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
            + [word[0] for word in topk_src_vocab_freq]
        )
        self.word2idSrc = {w: i for i, w in enumerate(self.vocabSrc)}
        self.id2wordSrc = {v: k for k, v in self.word2idSrc.items()}

        all_tgt_words = [word for sent in tgt for word in sent]
        topk_tgt_vocab_freq = Counter(all_tgt_words).most_common(self.num_tgt_vocab - 4)
        self.vocabTgt = np.array(
            ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
            + [word[0] for word in topk_tgt_vocab_freq]
        )
        self.word2idTgt = {w: i for i, w in enumerate(self.vocabTgt)}
        self.id2wordTgt = {v: k for k, v in self.word2idTgt.items()}

    def batched_ids2tokens(self, tokens, type="src"):
        """
        Converting sentence of ids to tokens

        :param tokens: Tokens Array, 2D array (num_samples, seq_len)
        :type tokens: numpy.ndarray
        :param type: {'src', 'tgt'} Type of tokens, defaults to "src"
        :type type: str, optional
        :return: List of decoded sentences
        :rtype: list
        """
        if type == "src":
            func = lambda x: self.id2wordSrc[x]
        else:
            func = lambda x: self.id2wordTgt[x]
        vect_func = np.vectorize(func)

        tokens = vect_func(tokens)

        sentences = []
        for words in tokens:
            txt = ""
            for word in words:
                if word not in ["<SOS>", "<EOS>", "<PAD>"]:
                    txt += f"{word} "
            sentences.append(txt[:-1])
        return sentences

    def preprocess_src(self, text):
        """
        Preprocessing Source Sentence

        :param text: Sentence
        :type text: str
        :return: Preprocessed sentence
        :rtype: str
        """
        text = text.lower().strip()
        text = re.sub(r"([?.!,¿_])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = re.sub(r"[^a-zA-Z?.!,¿_]+", " ", text)
        text = text.strip()

        return text.split()

    def preprocess_tgt(self, text):
        """
        Preprocessing Target Sentence

        :param text: Sentence
        :type text: str
        :return: Preprocessed sentence
        :rtype: str
        """
        text = "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "BN"
        )
        text = re.sub(r"([?.!,¿_])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = text.strip()

        return text.split()

    def extract_data(self):
        """
        Extracting Data from csv file

        :return: Train and Test DataFrames with Source and Target sentences
        :rtype: tuple, (pandas.DataFrame, pandas.DataFrame)
        """
        fpath = self.config_dict["paths"]["input_file"]

        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        src = [i.split("++++$++++")[0] for i in lines]
        tgt = [i.split("++++$++++")[1].rstrip() for i in lines]

        all_df = pd.DataFrame.from_dict({"Source": src, "Target": tgt})

        randomize = self.config_dict["preprocess"]["randomize"]
        num_samples = self.config_dict["dataset"]["num_samples"]
        test_split = self.config_dict["dataset"]["test_split"]
        num_test = int(num_samples * test_split)
        if randomize:
            ids = np.random.choice(len(all_df), num_samples, replace=False)
        else:
            ids = np.arange(num_samples)

        train_ids, test_ids = ids[:-num_test], ids[-num_test:]

        return all_df.iloc[train_ids], all_df.iloc[test_ids]
