import glob
import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from preprocess.utils import preprocess_text, BytePairEncoding


class PreprocessGPT:
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """        
        self.logger = logging.getLogger(__name__)

        self.input_folder = config_dict["paths"]["input_folder"]
        self.test_file = config_dict["paths"]["test_file"]
        self.random_lines = config_dict["dataset"]["random_lines"]
        self.num_setns_per_doc = config_dict["dataset"]["num_sents_per_doc"]
        self.test_samples = config_dict["dataset"]["test_samples"]
        self.seq_len = 1 + config_dict["dataset"]["seq_len"]
        self.operations = config_dict["preprocess"]["operations"]
        self.predict_tokens = config_dict["test"]["predict_tokens"]

        self.bpe = BytePairEncoding(config_dict)

    def get_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        text_ls = self.extract_data()
        text_ls = self.preprocess_text(text_ls)

        bpe_words = self.get_vocab(text_ls)
        text_tokens = np.zeros((len(text_ls), self.seq_len))

        text_lens = [len(text.split()) for text in text_ls]
        count = 0
        for i, text_len in enumerate(text_lens):
            text = " ".join(bpe_words[count: count+text_len]).split()
            count += text_len
            text = text[:self.seq_len]

            if len(text) < self.seq_len:
                text = text + ["<PAD>"]*(self.seq_len - len(text))

            text_tokens[i] = np.array([self.word2id[ch] for ch in text])

        return text_tokens
    
    def get_test_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        with open(self.test_file, "r") as f:
            lines = np.array(f.readlines())
        
        if self.random_lines:
            ids = np.random.choice(len(lines), self.num_setns_per_doc, replace=False)
        else:
            ids = np.arange(self.num_setns_per_doc)
        
        text_ls = lines[ids]
        text_ls = self.preprocess_text(text_ls)
        bpe_words = self.bpe.transform(text_ls)

        seq_len = self.seq_len - 1 + self.predict_tokens
        text_tokens = np.zeros((len(text_ls), seq_len))
        
        text_lens = [len(text.split()) for text in text_ls]
        count, num_sents = 0, 0
        for i, text_len in enumerate(text_lens):
            text = " ".join(bpe_words[count: count+text_len]).split()
            count += text_len
            text = text[:seq_len]

            if len(text) < seq_len:
                continue
            
            text_tokens[num_sents] = np.array([self.word2id[ch] if ch in self.word2id.keys() else self.word2id["<UNK>"] for ch in text])
            num_sents += 1

        self.logger.info(f"There are {num_sents} valid test Sentences")

        return text_tokens[:num_sents]


    def preprocess_text(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """        
        text_ls = [i.strip() for i in text_ls]
        text_ls = [preprocess_text(text, self.operations) for text in text_ls]

        return text_ls
    
    def get_vocab(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """        
        self.logger.info("Building Vocabulary using Byte Pair Encoding method")
        words = self.bpe.fit(text_ls)
        vocab = ["<PAD>", "<UNK>"] + list(self.bpe.vocab_freq.keys())

        self.word2id = {w:i for i, w in enumerate(vocab)}
        self.id2word = {v:k for k,v in self.word2id.items()}

        return words

    def batched_ids2tokens(self, tokens):
        """
        _summary_

        :param tokens: _description_
        :type tokens: _type_
        :return: _description_
        :rtype: _type_
        """        
        func = lambda x : self.id2word[x]
        vect_func = np.vectorize(func)

        tokens = vect_func(tokens)

        sentences = []
        for words in tokens:
            txt = ""
            for word in words:
                if word not in  ["<PAD>"]:
                    txt += f"{word} "
            txt  = txt[:-1]
            txt = txt.split("</w>")
            txt = " ".join([i.replace(" ", "") for i in txt])
            sentences.append(txt)
        
        return sentences
    
    def extract_data(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """        
        text_ls = []

        for path in glob.glob(f"{self.input_folder}\*.txt"):
            with open(path, "r") as f:
                lines = np.array(f.readlines())
            if self.random_lines:
                ids = np.random.choice(len(lines), self.num_setns_per_doc, replace=False)
            else:
                ids = np.arange(self.num_setns_per_doc)
                
            lines = lines[ids]
            text_ls.extend(lines)

        return text_ls
    

def create_dataloader(X, data="train", val_split=0.2, batch_size=32, seed=2024): 
    """
    _summary_

    :param X: _description_
    :type X: _type_
    :param data: _description_, defaults to "train"
    :type data: str, optional
    :param val_split: _description_, defaults to 0.2
    :type val_split: float, optional
    :param batch_size: _description_, defaults to 32
    :type batch_size: int, optional
    :param seed: _description_, defaults to 2024
    :type seed: int, optional
    :return: _description_
    :rtype: _type_
    """    
    if data == "train":
        train_X, val_X = train_test_split(X, test_size=val_split, random_state=seed)

        train_ds = TensorDataset(torch.Tensor(train_X))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

        val_ds = TensorDataset(torch.Tensor(val_X))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)

        return train_loader, val_loader
    else:
        test_ds = TensorDataset(torch.Tensor(X))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)
    
        return test_loader