import os
import nltk
import itertools
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split    

from .utils import viz_metrics, plot_hist_dataset, plot_transition_matrix, pca_emission_matrix

class HMM():
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """        
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

    def run(self):
        """
        _summary_
        """        
        trainX, testX, trainY, testY = self.preprocess()
        self.fit(trainX, trainY)
        testYP = self.predict(testX)
        metric_dict = self.evaluate(testY, testYP)
        self.save_output((trainX, testX, trainY, testY), metric_dict)

    def fit(self, X, y):
        """
        _summary_

        :param X: _description_
        :type X: _type_
        :param y: _description_
        :type y: _type_
        """        
        if X is None or y is None:
            self.logger.error("X/&y is/are missing in fit() function")

        self.x_vocab = list(set(itertools.chain.from_iterable(X)))
        self.x_vocab_dict = {k:v for v,k in enumerate(self.x_vocab)}
        self.y_vocab = list(set(itertools.chain.from_iterable(y)))
        self.y_vocab_dict = {k:v for v,k in enumerate(self.y_vocab)}

        num_x = len(self.x_vocab)
        num_y = len(self.y_vocab)
        
        # p(t/w) = p(w/t)*p(t)/p(w)
        self.trans_matrix = np.zeros((num_y, num_y))
        self.em_matrix = np.zeros((num_y, num_x))

        for pos_sent in y:
            for i in range(len(pos_sent)-1):
                pos_t0, pos_t1 = pos_sent[i], pos_sent[i+1]
                self.trans_matrix[self.y_vocab_dict[pos_t0], self.y_vocab_dict[pos_t1]] += 1
        self.trans_matrix = self.trans_matrix/np.sum(self.trans_matrix, axis=0)

        for sent, pos_sent in zip(X,y):
            for i in range(len(sent)):
                t, pos_t = sent[i], pos_sent[i]
                self.em_matrix[self.y_vocab_dict[pos_t], self.x_vocab_dict[t]] += 1
        self.em_matrix = self.em_matrix/np.sum(self.em_matrix, axis=1, keepdims=True)


    def predict(self, X):
        """
        _summary_

        :param X: _description_
        :type X: _type_
        :return: _description_
        :rtype: _type_
        """        
        if X is None:
            self.logger.error("X is missing in predict() function")
        y_pred_total = []

        for sent in X:
            y_pred_sent = []
            for i, word in enumerate(sent):
                probs = np.zeros((len(self.y_vocab)))
                for j, pos_ in enumerate(self.y_vocab):
                    if  i == 0:
                        trans_p = self.trans_matrix[self.y_vocab_dict["."], self.y_vocab_dict[pos_]]
                    else:
                        trans_p = self.trans_matrix[self.y_vocab_dict[y_pred_sent[-1]], self.y_vocab_dict[pos_]]
                    
                    if word in self.x_vocab:
                        em_p = self.em_matrix[self.y_vocab_dict[pos_], self.x_vocab_dict[word]]
                    else:
                        em_p = 1/len(self.y_vocab)

                    probs[j] = trans_p*em_p
                y_pred_sent.append(self.y_vocab[np.argmax(probs)])

            y_pred_total.append(y_pred_sent)

        return y_pred_total
                

    def evaluate(self, Y, YP):
        """
        _summary_

        :param Y: _description_
        :type Y: _type_
        :param YP: _description_
        :type YP: _type_
        :return: _description_
        :rtype: _type_
        """        
        metric_dict = {}

        y_true = list(itertools.chain.from_iterable(Y))
        y_pred = list(itertools.chain.from_iterable(YP))

        metric_dict["clf_report"] = classification_report(y_true, y_pred, labels=self.y_vocab, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.y_vocab)
        metric_dict["conf_matrix"] = pd.DataFrame(conf_matrix, index=self.y_vocab, columns=self.y_vocab)
        return metric_dict

    def preprocess(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        num_samples = self.config_dict["dataset"]["num_samples"]
        seed = self.config_dict["dataset"]["seed"]
        test_size = self.config_dict["dataset"]["test_size"]

        sents = list(nltk.corpus.treebank.tagged_sents())[:num_samples]
        words = [[i for i,j in k] for k in sents]
        pos = [[j for i,j in k] for k in sents]
        trainX, testX, trainY, testY = train_test_split(words, pos, test_size=test_size, random_state=seed)

        return trainX, testX, trainY, testY 


    def save_output(self, data, metric_dict):
        """
        _summary_

        :param data: _description_
        :type data: _type_
        :param metric_dict: _description_
        :type metric_dict: _type_
        """        
        self.logger.info("Saving Vectors and Plots into Output Folder")
        output_folder = self.config_dict["paths"]["output_folder"]
        os.makedirs(output_folder, exist_ok=True)
        visualize = self.config_dict["visualize"]

        trans_matrix_df = pd.DataFrame(self.trans_matrix, columns=self.y_vocab, index=self.y_vocab)
        em_matrix_df = pd.DataFrame(self.em_matrix, index=self.y_vocab, columns=self.x_vocab)

        trans_matrix_df.to_csv(os.path.join(output_folder, "Transition Matrix.csv"))
        em_matrix_df.to_csv(os.path.join(output_folder, "Emission Matrix.csv"))

        if visualize:
            viz_metrics(metric_dict, output_folder)
            plot_hist_dataset(data, output_folder)
            plot_transition_matrix(trans_matrix_df, output_folder)
            pca_emission_matrix(em_matrix_df, output_folder)