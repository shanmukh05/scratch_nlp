import torch
import itertools
import numpy as np

class TextGenerationMetrics():
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.rouge_n_n = config_dict["train"]["rouge_n_n"]
        self.rouge_s_n = config_dict["train"]["rouge_s_n"]
        self.bleu_n = config_dict["train"]["bleu_n"]
        self.seq_len = config_dict["dataset"]["seq_len"]

    def get_metrics(self, references, predictions):
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

    def bleu_score(self, references, candidates, n=4):
        return 0.5

    def perplexity_score(self, predictions):
        predictions = np.max(predictions, axis=-1)
        probs_prod = np.prod(predictions, axis=1)
        probs_prod_inv_norm = 1/np.pow(probs_prod, 1/self.seq_len)

        score = np.mean(probs_prod_inv_norm).item()
        return score

    def meteor_score(self, references, predictions):
        predictions = np.argmax(predictions, axis=-1)
        return 0.5

    def rouge_n_score(self, references, predictions, n=4):
        predictions = np.argmax(predictions, axis=-1)
        num_instances = references.shape[0]
        precision, recall, f1score = 0, 0, 0

        for ref, pred in zip(references, predictions):
            ref_n = [" ".join([str(ref[j+k]) for k in range(n)]) for j in range(len(ref)-n+1)]
            pred_n = [" ".join([str(pred[j+k]) for k in range(n)]) for j in range(len(pred)-n+1)]

            n_unq_ngrams = len(set(ref_n) & set(pred_n))
            p, r = n_unq_ngrams/len(pred_n), n_unq_ngrams/len(ref_n)
            precision += p
            recall += r
            f1score += 2*p*r/(p + r + 1e-8)

        return precision/num_instances, recall/num_instances, f1score/num_instances


    def rouge_l_score(self, references, predictions):
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
        return 0.5
    
    def _lcs(self, arr1, arr2):
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

