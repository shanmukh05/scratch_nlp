import cv2
import torch
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess.utils import preprocess_text, WordPiece


class PreprocessBERT:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.input_path = config_dict["paths"]["input_file"]
        self.num_samples = config_dict["dataset"]["num_samples"]
        self.seq_len = config_dict["dataset"]["seq_len"]
        self.operations = config_dict["preprocess"]["operations"]

        self.wordpiece = WordPiece(config_dict)

    def  get_data(self):
        text_ls = self.extract_data()
        text_ls = self.preprocess_text(text_ls)
        text_lens = [len(text.split()) for text in text_ls]

        corpus = self.get_vocab(text_ls)

        half_seq_len = self.seq_len//2 - 1
        text_tokens_a = np.zeros((len(text_ls), half_seq_len+1))
        text_tokens_b = np.zeros((len(text_ls), half_seq_len+1))
        count = 0
        for i, text_len in enumerate(text_lens):
            tokens_ls = corpus[count: count+text_len]
            tokens = [i for ls in tokens_ls for i in ls]
            text = " ".join(tokens).split()
            count += text_len

            text = text[:self.seq_len]

            if len(text) < self.seq_len:
                text = text + ["<PAD>"]*(self.seq_len - len(text))

            text_tokens_a[i] = np.array([self.word2id[ch] for ch in ["<CLS>"] + text[:half_seq_len]])
            text_tokens_b[i] = np.array([self.word2id[ch] for ch in ["<SEP>"] + text[half_seq_len:2*half_seq_len]])

        reorder_tokens_b = np.random.choice(len(text_ls), len(text_ls), replace=False)
        text_tokens_a = np.concatenate([text_tokens_a, text_tokens_a], axis=0)
        text_tokens_b = np.concatenate([text_tokens_b, text_tokens_b[reorder_tokens_b]], axis=0)
        text_tokens = np.concatenate([text_tokens_a, text_tokens_b], axis=-1)
        
        nsp_labels = np.array(["IsNext"]*len(text_ls) + ["NotNext"]*len(text_ls))
        self.lencoder = LabelEncoder()
        nsp_labels = self.lencoder.fit_transform(nsp_labels)

        return text_tokens, nsp_labels


    def preprocess_text(self, text_ls):
        text_ls = [preprocess_text(text, self.operations) for text in text_ls]

        return text_ls
    
    def get_vocab(self, text_ls):
        self.logger.info("Building Vocabulary using Byte Pair Encoding method")
        corpus = self.wordpiece.fit(text_ls)
        vocab = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"] + list(self.wordpiece.vocab_freq.keys())

        self.word2id = {w:i for i, w in enumerate(vocab)}
        self.id2word = {v:k for k,v in self.word2id.items()}

        return corpus
    
    def batched_ids2tokens(self, tokens):
        func = lambda x : self.id2word[x]
        vect_func = np.vectorize(func)

        tokens = vect_func(tokens)

        sentences = []
        for seq in tokens:
            start_id = 0
            words = []
            for i, ch in enumerate(seq):
                if "##" != ch[:2] and i != 0:
                    tokens = seq[start_id:i]
                    word = "".join([w if i==0 else w[2:] for i,w in enumerate(tokens)])
                    words.append(word)
                    start_id = i
            final_word = "".join([w if i==0 else w[2:] for i,w in enumerate(seq[start_id:])])
            words.append(final_word)
            sentences.append(" ".join(words))
        
        return sentences

    def extract_data(self):
        df = pd.read_csv(self.input_path, nrows=self.num_samples)

        return df["text"]


class BERTDataset(Dataset):
    def __init__(self, text_tokens, nsp_labels, config_dict, word2id):
        self.text_tokens = text_tokens
        self.nsp_labels = nsp_labels
        self.word2id = word2id

        self.num_vocab = config_dict["dataset"]["num_vocab"]
        self.seq_len = config_dict["dataset"]["seq_len"]
        self.half_seq_len = self.seq_len//2 - 1

        pred_prob = config_dict["preprocess"]["replace_token"]["prediction"]
        pred_mask = config_dict["preprocess"]["replace_token"]["mask"]
        pred_random = config_dict["preprocess"]["replace_token"]["random"]
        self.num_extra_tokens = config_dict["dataset"]["num_extra_tokens"]

        self.num_pred_tokens_half = int(pred_prob*self.seq_len)
        self.num_mask_tokens = int(2*self.num_pred_tokens_half*pred_mask)
        self.num_rand_tokens = int(2*self.num_pred_tokens_half*pred_random)

    def __len__(self):
        return len(self.text_tokens)
    
    def __getitem__(self, idx):
        nsp_label = self.nsp_labels[idx]
        text_token = self.text_tokens[idx].to(torch.int64)
        text_token, lbl_mask = self._apply_mask(text_token)
        
        return text_token, lbl_mask, nsp_label
    
    def _apply_mask(self, text_token):
        lbl_mask_ids_a = 1 + torch.randperm(self.half_seq_len)[:self.num_pred_tokens_half]
        lbl_mask_ids_b = 1 + self.seq_len//2 + torch.randperm(self.half_seq_len)[:self.num_pred_tokens_half]
        lbl_mask_ids = torch.concat([lbl_mask_ids_a, lbl_mask_ids_b], axis=0)
        lbl_mask = torch.zeros_like(text_token)
        lbl_mask[lbl_mask_ids] = 1

        rand_tokens = torch.randperm(self.num_vocab-self.num_extra_tokens)[:self.num_rand_tokens] + self.num_extra_tokens
        text_token[lbl_mask_ids[:self.num_mask_tokens]] = self.word2id["<MASK>"]
        text_token[lbl_mask_ids[self.num_mask_tokens:self.num_mask_tokens+self.num_rand_tokens]] = rand_tokens

        return text_token, lbl_mask  


def create_dataloader(X, y, config_dict, word2id, val_split=0.2, test_split=0.2, batch_size=32, seed=2024): 
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=val_split+test_split, random_state=seed)
    val_X, test_X, val_y, test_y = train_test_split(X, y, test_size=test_split/(val_split+test_split), random_state=seed)

    train_ds = BERTDataset(torch.Tensor(train_X), torch.Tensor(train_y), config_dict, word2id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

    val_ds = BERTDataset(torch.Tensor(val_X), torch.Tensor(val_y), config_dict, word2id)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)
        
    test_ds = BERTDataset(torch.Tensor(test_X), torch.Tensor(test_y), config_dict, word2id)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)
    
    return train_loader, val_loader, test_loader