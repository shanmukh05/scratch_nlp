import os
import numpy as np
import pandas as pd
import logging

from core.bow.bow import BOW
from plot_utils import plot_pca_pairplot


class TFIDF(BOW):
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    """
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict
        self.return_label = self.config_dict["model"]["output_label"]

    def run(self):
        """
        _summary_
        """
        self.preprocess()
        X, y = self.fit_transform()
        self.save_output(X, y)

    def fit(self, text_ls=None, y=None):
        """
        _summary_

        :param text_ls: _description_, defaults to None
        :type text_ls: _type_, optional
        :param y: _description_, defaults to None
        :type y: _type_, optional
        """
        self.logger.info("Fitting TF-IDF to Extracted Data")
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        self.vocab = []
        for text in text_ls:
            uniq_words = list(set(text.split()))
            self.vocab.extend(uniq_words)
            self.vocab = list(set(self.vocab))
        self.vocab_dict = {k: v for v, k in enumerate(self.vocab)}

    def fit_transform(self, text_ls=None, y=None):
        """
        _summary_

        :param text_ls: _description_, defaults to None
        :type text_ls: _type_, optional
        :param y: _description_, defaults to None
        :type y: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        if text_ls is None:
            text_ls, y = self.text_ls, self.y
        self.fit(text_ls, y)
        X, y = self.transform(text_ls, y)
        return X, y

    def transform(self, text_ls=None, y=None):
        """
        _summary_

        :param text_ls: _description_, defaults to None
        :type text_ls: _type_, optional
        :param y: _description_, defaults to None
        :type y: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        self.logger.info("Transforming Txt Data into Vectors")
        if text_ls is None:
            text_ls, y = self.text_ls, self.y

        self.tf_arr = self.get_tf(text_ls)
        self.idf_arr = self.get_idf(text_ls)

        X = (self.tf_arr.T) * (self.idf_arr.T)
        return X, y

    def get_tf(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        """
            Binary
            Raw Count
            Term Frequency
            Log Norm
            Double Norm
        """
        self.logger.info("Calculating TF")
        tf_mode = self.config_dict["model"]["tf_mode"]
        tf_arr = np.zeros((len(self.vocab), len(text_ls)))

        for i, text in enumerate(text_ls):
            for word in text.split():
                tf_arr[self.vocab_dict[word]][i] += 1

        if tf_mode == "binary":
            tf_arr = np.where(tf_arr > 0, 1, 0)
        elif tf_mode == "raw_count":
            tf_arr = tf_arr
        elif tf_mode == "term_frequency":
            tf_arr = tf_arr / np.sum(tf_arr, axis=1)
        elif tf_mode == "log_norm":
            tf_arr = np.log(1 + tf_arr)
        elif tf_mode == "double_norm":
            tf_arr = 0.5 + 0.5 * tf_arr / np.max(tf_arr, axis=1)
        else:
            self.logger.error("Invalid TF Mode.")

        return tf_arr

    def get_idf(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        """
            Unary
            Log Scaled
            Log Scaled Smoothing
            Log Scaled Max
            Log Scaled Probablistic
        """
        self.logger.info("Calculating IDF")
        idf_mode = self.config_dict["model"]["idf_mode"]

        tf_binary = np.where(self.tf_arr > 0, 1, 0)
        idf_arr = np.sum(tf_binary, axis=1)
        N = len(text_ls)

        if idf_mode == "unary":
            idf_arr = np.where(idf_arr > 0, 1, 0)
        elif idf_mode == "log_scaled":
            idf_arr = np.log(N / idf_arr)
        elif idf_mode == "log_scaled_smoothing":
            idf_arr = 1 + np.log(N / (1 + idf_arr))
        elif idf_mode == "log_scaled_max":
            idf_arr = np.log(np.max(idf_arr) / (1 + idf_arr))
        elif idf_mode == "log_scaled_probablistic":
            idf_arr = np.log(N / idf_arr - 1)
        else:
            self.logger.error("Invalid IDF Mode.")

        return idf_arr

    def save_output(self, X, y):
        """
        _summary_

        :param X: _description_
        :type X: _type_
        :param y: _description_
        :type y: _type_
        """
        self.logger.info("Saving Vectors and Plots into Output Folder")
        output_folder = self.config_dict["paths"]["output_folder"]
        os.makedirs(output_folder, exist_ok=True)
        visualize = self.config_dict["visualize"]

        np.save(os.path.join(output_folder, "Text Vector.npy"), X)
        np.save(os.path.join(output_folder, "Text Label.npy"), y)

        if visualize:
            plot_pca_pairplot(X, y, output_folder)
            plot_pca_pairplot(self.tf_arr.T, y, output_folder, name="TF PCA Pairplot")
            idf_df = pd.DataFrame.from_dict({"Token": self.vocab, "IDF": self.idf_arr})
            idf_df.to_csv(os.path.join(output_folder, "IDF.csv"), index=False)
