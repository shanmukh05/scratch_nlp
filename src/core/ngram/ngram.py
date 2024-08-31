import os
import numpy as np
import pandas as pd
import logging

from core.bow.bow import BOWModel
from core.bow.utils import plot_topk_freq, plot_wordcloud
from .utils import plot_pie_chart


class NGRAMModel(BOWModel):
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict
        self.return_label = self.config_dict["model"]["output_label"]
        self.n = self.config_dict["model"]["ngram"]

    def run(self):
        self.preprocess()
        X, y = self.fit_transform()
        self.save_output(X, y)

    def fit(self, text_ls=None, y=None):
        self.logger.info("Fitting NGRAM to Extracted Data")
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        self.vocab = []
        for text in text_ls:
            tokens = text.split()
            uniq_words = [" ".join([tokens[j+k] for k in range(self.n)]) for j in range(len(tokens)-self.n)]
            self.vocab.extend(uniq_words)
            self.vocab = list(set(self.vocab))
    
    def fit_transform(self, text_ls=None, y=None):
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        self.fit(text_ls, y)
        X, y = self.transform(text_ls, y)
        return X, y
    
    def transform(self, text_ls=None, y=None):
        self.logger.info("Transforming Txt Data into Vectors")
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        X = np.zeros((len(text_ls), len(self.vocab)))
        self.vocab_freq = {k: v for k, v in zip(self.vocab, [0] * len(self.vocab))}

        for i, text in enumerate(text_ls):
            count_dict = {k: v for k, v in zip(self.vocab, [0] * len(self.vocab))}
            tokens = text.split()
            for j in range(len(tokens)-self.n):
                word = " ".join([tokens[j+k] for k in range(self.n)])
                count_dict[word] += 1
                self.vocab_freq[word] += 1
            X[i] = list(count_dict.values())
        return X, y
    
    def save_output(self, X, y):
        self.logger.info("Saving Vectors and Plots into Output Folder")
        output_folder = self.config_dict["paths"]["output_folder"]
        os.makedirs(output_folder, exist_ok=True)
        visualize = self.config_dict["visualize"]

        np.save(os.path.join(output_folder, "Text Vector.npy"), X)
        np.save(os.path.join(output_folder, "Text Label.npy"), y)

        df = pd.DataFrame.from_dict(
            {"Word": self.vocab_freq.keys(), "Frequency": self.vocab_freq.values()}
        )
        df.to_csv(os.path.join(output_folder, "Vocab Frequency.csv"), index=False)

        if visualize:
            plot_topk_freq(self.vocab_freq, output_folder)
            plot_wordcloud(self.vocab_freq, output_folder)
            plot_pie_chart(df, self.n, output_folder)


