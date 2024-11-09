import itertools
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report

class ClassificationMetrics():
    """
    Metrics for Classification Task. 

    :param config_dict: Config Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def get_metrics(self, references, predictions, target_names):
        """
        Function that returns Metrics using References, Predictions and Class Labels

        :param references: References, 1D array (N,)
        :type references: numpy.array
        :param predictions: Predictions, 2D array (NxC) with Probabilities 
        :type predictions: numpy.array
        :param target_names: Class Labels
        :type target_names: list
        :return: Metrics Dictionary
        :rtype: dict
        """
        predictions = np.argmax(predictions, axis=-1)

        labels = np.arange(len(target_names))
        clf_report = classification_report(references, predictions, labels=labels, target_names=target_names, output_dict=True, zero_division=0.0)
        metric_dict = {}

        metric_dict["Accuracy"] = clf_report["accuracy"]
        for name in target_names + ["macro avg", "weighted avg"]:
            metric_dict[f"{name}-Precision"] = clf_report[name]["precision"]
            metric_dict[f"{name}-Recall"] = clf_report[name]["recall"]
            metric_dict[f"{name}-F1Score"] = clf_report[name]["f1-score"]

        return metric_dict


class TextGenerationMetrics():
    """
    Metrics for Text Generation Task. 

    :param config_dict: Config Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.rouge_n_n = config_dict["train"]["rouge_n_n"]
        self.rouge_s_n = config_dict["train"]["rouge_s_n"]
        self.bleu_n = config_dict["train"]["bleu_n"]
        self.seq_len = config_dict["dataset"]["seq_len"]

    def get_metrics(self, references, predictions):
        """
        Function that returns Metrics using References, Predictions and Class Labels

        :param references: References, 2D array (N,S)
        :type references: numpy.array
        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :return: Metrics Dictionary
        :rtype: dict
        """
        rouge_n_p, rouge_n_r, rouge_n_f = self.rouge_n_score(references, predictions, self.rouge_n_n)
        rouge_l_p, rouge_l_r, rouge_l_f = self.rouge_l_score(references, predictions)
        rouge_s_p, rouge_s_r, rouge_s_f = self.rouge_s_score(references, predictions, self.rouge_s_n)

        return {
            f"BLEU-{self.bleu_n}": self.bleu_score(references, predictions, self.bleu_n),
            "Perplexity": self.perplexity_score(predictions),
            "METEOR": self.meteor_score(references, predictions),
            f"ROUGE-N-{self.rouge_n_n}-Precision": rouge_n_p,
            f"ROUGE-N-{self.rouge_n_n}-Recall": rouge_n_r,
            f"ROUGE-N-{self.rouge_n_n}-F1score": rouge_n_f,
            "ROUGE-L-Precision": rouge_l_p,
            "ROUGE-L-Recall": rouge_l_r,
            "ROUGE-L-F1score": rouge_l_f,
            f"ROUGE-S-{self.rouge_s_n}-Precision": rouge_s_p,
            f"ROUGE-S-{self.rouge_s_n}-Recall": rouge_s_r,
            f"ROUGE-S-{self.rouge_s_n}-F1score": rouge_s_f,
            "CIDER": self.cider_score(references, predictions)
        }

    def bleu_score(self, references, predictions, n=4):
        """
        BLEU Score

        :param references: References, 2D array (N,S)
        :type references: numpy.array
        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :param n: Max number of N gram, defaults to 4
        :type n: int, optional
        :return: BLEU score
        :rtype: float
        """
        predictions = np.argmax(predictions, axis=-1)
        num_instances = references.shape[0]
        log_score_corpus = 0
        w = 1/n

        for ref, pred in zip(references, predictions):
            ref_len, pred_len = len(ref), len(pred)
            bp = min(0, 1-ref_len/pred_len)
            log_score = bp

            for i in range(1, n+1):
                p, _, _ = self._get_metrics_ngram(ref, pred, i, True)
                log_score += w*np.log(1 + p)

            log_score_corpus += log_score/num_instances
                
        return log_score_corpus

    def perplexity_score(self, predictions):
        """
        Perplixity Score

        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :return: Perplixity Score
        :rtype: float
        """
        predictions = np.max(predictions, axis=-1)
        probs_sum = -np.sum(np.log(predictions), axis=1)/self.seq_len
        probs_prod_inv_norm = np.exp(probs_sum)

        # probs_prod = np.prod(predictions, axis=1)
        # probs_prod_inv_norm = 1/np.pow(probs_prod, 1/self.seq_len)

        score = np.mean(probs_prod_inv_norm).item()
        return score

    def meteor_score(self, references, predictions):
        predictions = np.argmax(predictions, axis=-1)
        return 0.5

    def rouge_n_score(self, references, predictions, n=4):
        """
        ROUGE N Score

        :param references: References, 3D array (N,S)
        :type references: numpy.array
        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :param n: Max number of N gram, defaults to 4
        :type n: int, optional
        :return: ROUGE N score
        :rtype: float
        """
        predictions = np.argmax(predictions, axis=-1)
        num_instances = references.shape[0]
        precision, recall, f1score = 0, 0, 0

        for ref, pred in zip(references, predictions):
            p, r, f = self._get_metrics_ngram(ref, pred, n)
            precision += p
            recall += r
            f1score += f

        return precision/num_instances, recall/num_instances, f1score/num_instances


    def rouge_l_score(self, references, predictions):
        """
        ROUGE L Score

        :param references: References, 2D array (N,S)
        :type references: numpy.array
        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :return: ROUGE L score
        :rtype: float
        """
        predictions = np.argmax(predictions, axis=-1)
        num_instances = references.shape[0]
        precision, recall, f1score = 0, 0, 0

        for ref, pred in zip(references, predictions):
            lcs = self._lcs(ref, pred)
            p, r = lcs/len(pred), lcs/len(ref)
            precision += p
            recall += p
            f1score += 2*p*r/(p + r + 1e-8)

        return precision/num_instances, recall/num_instances, f1score/num_instances

    
    def rouge_s_score(self, references, predictions, n=4):
        """
        ROUGE S Score

        :param references: References, 2D array (N,S)
        :type references: numpy.array
        :param predictions: Predictions, 3D array (NxSxC) with Probabilities 
        :type predictions: numpy.array
        :param n: Max number of N gram, defaults to 4
        :type n: int, optional
        :return: ROUGE S score
        :rtype: float
        """
        predictions = np.argmax(predictions, axis=-1)
        num_instances = references.shape[0]
        precision, recall, f1score = 0, 0, 0

        for ref, pred in zip(references, predictions):
            ref_bi = [" ".join([str(ref[j]), str(ref[j+1])]) for j in range(len(ref)-1)]
            pred_skip_bi = [[" ".join([str(pred[i]), str(pred[k])]) for k in range(i+1, min(len(pred)-1, i+n+1))] for i in range(len(pred)-1)]
            pred_skip_bi =  list(itertools.chain.from_iterable(pred_skip_bi))

            n_unq_ngrams = len(set(ref_bi) & set(pred_skip_bi))
            p, r = n_unq_ngrams/len(pred_skip_bi), n_unq_ngrams/len(ref_bi)
            precision += p
            recall += r
            f1score += 2*p*r/(p + r + 1e-8)

        return precision/num_instances, recall/num_instances, f1score/num_instances

    def cider_score(self, references, predictions):
        predictions = np.argmax(predictions, axis=-1)
        return 0.5

    def _get_metrics_ngram(self, ref, pred, n, clip=False):
        """
        Obtain Precision, Recall and F1 score of NGRAM

        :param ref: Reference Sentence, 1D Array (S,)
        :type ref: numpy.array
        :param pred: Predicted Sentence, 1D ARray (S,)
        :type pred: numpy.array
        :param n: Number of tokens in a base token
        :type n: int
        :param clip: _description_, defaults to False
        :type clip: bool, optional
        :return: Precision, Recall, F1 score
        :rtype: float, float, float
        """
        ref_n = [" ".join([str(ref[j+k]) for k in range(n)]) for j in range(len(ref)-n+1)]
        pred_n = [" ".join([str(pred[j+k]) for k in range(n)]) for j in range(len(pred)-n+1)]
        
        if clip:
            n_common_ngrams = 0
            ref_cnt_dict = Counter(ref)
            pred_cnt_dict = Counter(pred)
            for k in pred_cnt_dict.keys():
                cnt_k = min(pred_cnt_dict[k], ref_cnt_dict.get(k, 0))
                n_common_ngrams += cnt_k
        else:
            n_common_ngrams = len(set(ref_n) & set(pred_n))
        p, r = n_common_ngrams/len(pred_n), n_common_ngrams/len(ref_n)
        f = 2*p*r/(p + r + 1e-8)

        return p, r, f

    
    def _lcs(self, arr1, arr2):
        """
        Find Longest Common Subsequence 

        :param arr1: First Array, 1D Array
        :type arr1: numpy.array
        :param arr2: Second Array, 1D Array
        :type arr2: numpy.array
        :return: Longest Common Subsequence Length
        :rtype: int
        """
        m = len(arr1)
        n = len(arr2)

        lcs_mat = np.zeros((m+1, n+1))

        for i in range(1, m+1):
            for j in range(1, n+1):
                if arr1[i-1] == arr2[j-1]:
                    lcs_mat[i][j] = lcs_mat[i-1][j-1] + 1
                else:
                    lcs_mat[i][j] = max(lcs_mat[i-1][j], lcs_mat[i][j-1])

        return float(lcs_mat[m][n])

