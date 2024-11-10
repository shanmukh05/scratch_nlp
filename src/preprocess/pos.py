import logging
import numpy as np
from nltk.corpus import *
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

from .utils import preprocess_text

import nltk
nltk.download('treebank')
nltk.download("brown")
nltk.download("conll2000")


CORPUS = {
    "treebank": treebank.tagged_sents(tagset="universal"),
    "brown": brown.tagged_sents(tagset="universal"),
    "con11": conll2000.tagged_sents(tagset="universal"),
}


class PreprocessPOS:
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """
        self.logger = logging.getLogger(__name__)

        self.config_dict = config_dict
        self.operations = config_dict["preprocess"]["operations"]
        self.num_vocab = config_dict["dataset"]["num_vocab"]
        self.seq_len = config_dict["dataset"]["seq_len"]

        self.label_encoder = OneHotEncoder()

        self.corpus, self.test_corpus = self.extract_data()
        self.get_vocab(self.corpus)

    def get_data(self, corpus):
        """
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        :return: _description_
        :rtype: _type_
        """
        X, y = self.preprocess_corpus(corpus)

        tokenX = np.zeros((len(X), self.seq_len))
        labels = np.zeros((len(y), self.seq_len, len(self.unq_pos)))
        for i, (sent, sent_pos) in enumerate(zip(X, y)):
            sent = sent[: self.seq_len]
            sent_pos = sent_pos[: self.seq_len]
            sent_labels = np.zeros((self.seq_len))
            if len(sent) < self.seq_len:
                sent = sent + ["<PAD>"] * (self.seq_len - len(sent))
                sent_pos = sent_pos + ["<PAD>"] * (self.seq_len - len(sent))
            for j, (word, word_pos) in enumerate(zip(sent, sent_pos)):
                bool_ = word in self.vocabX
                tokenX[i][j] = self.word2idX[word] if bool_ else self.word2idX["<UNK>"]
                sent_labels[j] = (
                    self.posEnc[word_pos] if bool_ else self.posEnc["<UNK>"]
                )
            labels[i] = self.label_encoder.transform(
                sent_labels.reshape(-1, 1)
            ).toarray()

        return tokenX, labels

    def get_vocab(self, corpus):
        """
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        """
        self.logger.info(
            "Building Vocabulary for Words and POS tags from training data"
        )
        X, y = self.preprocess_corpus(corpus)

        all_words = [word for sent in X for word in sent]
        topk_vocab_freq = Counter(all_words).most_common(self.num_vocab - 2)

        self.vocabX = np.array(
            ["<PAD>", "<UNK>"] + [word[0] for word in topk_vocab_freq]
        )
        self.unq_pos = np.array(
            ["<PAD>", "<UNK>"]
            + list(set([word_pos for sent_pos in y for word_pos in sent_pos]))
        )

        self.word2idX = {w: i for i, w in enumerate(self.vocabX)}
        self.id2wordX = {v: k for k, v in self.word2idX.items()}

        self.posEnc = {w: i for i, w in enumerate(self.unq_pos)}
        self.posDec = {v: k for k, v in self.posEnc.items()}

        self.label_encoder.fit(np.arange(len(self.unq_pos)).reshape(-1, 1))

    def extract_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """
        self.logger.info("Extracting Train and Test corpus from global CORPUS variable")
        corpus = []
        for name in self.config_dict["dataset"]["train_corpus"]:
            corpus += CORPUS[name]

        test_corpus = []
        for name in self.config_dict["dataset"]["test_corpus"]:
            test_corpus += CORPUS[name]

        randomize = self.config_dict["preprocess"]["randomize"]
        num_train = self.config_dict["dataset"]["train_samples"]
        num_test = self.config_dict["dataset"]["test_samples"]
        if randomize:
            train_ids = np.random.choice(len(corpus), num_train, replace=False)
            test_ids = np.random.choice(len(test_corpus), num_test, replace=False)
        else:
            train_ids = np.arange(num_train)
            test_ids = np.arange(num_test)

        return [corpus[i] for i in train_ids], [test_corpus[i] for i in test_ids]

    def preprocess_corpus(self, corpus):
        """
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        :return: _description_
        :rtype: _type_
        """
        X = [[i[0] for i in sent] for sent in corpus]
        y = [[i[1] for i in sent] for sent in corpus]

        X = [preprocess_text(" ".join(sent), self.operations).split() for sent in X]

        return X, y
