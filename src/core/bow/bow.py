import os
import numpy as np
import pandas as pd
import logging
from preprocess.imdb_reviews import PreprocessIMDB
from plot_utils import plot_topk_freq, plot_wordcloud


class BOW:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict
        self.return_label = self.config_dict["model"]["output_label"]

    def run(self):
        self.preprocess()
        X, y = self.fit_transform()
        self.save_output(X, y)

    def fit(self, text_ls=None, y=None):
        self.logger.info("Fitting BOW to Extracted Data")
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        self.vocab = []
        for text in text_ls:
            uniq_words = list(set(text.split()))
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
            for word in text.split():
                count_dict[word] += 1
                self.vocab_freq[word] += 1
            X[i] = list(count_dict.values())
        return X, y

    def preprocess(self):
        root_path = self.config_dict["paths"]["input_folder"]
        explore_folder = self.config_dict["dataset"]["explore_folder"]
        num_samples = self.config_dict["dataset"]["num_samples"]
        operations = self.config_dict["preprocess"]["operations"]
        randomize = self.config_dict["preprocess"]["randomize"]

        self.preproc_cls = PreprocessIMDB(
            root_path, explore_folder, num_samples, operations, randomize
        )
        self.preproc_cls.run()
        self.text_ls, self.y = self.preproc_cls.text_ls, self.preproc_cls.label_ls
        self.logger.info("Preprocessing Done")

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
