import json
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess.utils import preprocess_text


class PreprocessBERTFinetune:
    def __init__(self, config_dict, wordpiece, word2id):
        self.logger = logging.getLogger(__name__)

        self.input_path = config_dict["finetune"]["paths"]["input_file"]
        self.num_topics = config_dict["finetune"]["dataset"]["num_topics"]
        self.seq_len = config_dict["dataset"]["seq_len"]
        self.operations = config_dict["preprocess"]["operations"]

        self.wordpiece = wordpiece
        self.word2id = word2id
        self.id2word = {v:k for k,v in self.word2id.items()}

    def get_data(self):
        df = self.extract_data()

        df["Context"] = df["Context"].map(lambda x: self.preprocess_text(x))
        df["Question"] = df["Question"].map(lambda x: self.preprocess_text(x))

        vocab = self.word2id.keys()

        half_seq_len = self.seq_len//2 - 1

        ques_tokens, cxt_tokens = [], []
        start_ids, end_ids = [], []
        topics = []

        cxt_lens = [len(text.split()) for text in df["Context"]]
        ques_lens = [len(text.split()) for text in df["Question"]]

        cxt_corpus = self.wordpiece.transform(df["Context"])
        ques_corpus = self.wordpiece.transform(df["Question"])

        cxt_count, ques_count = 0, 0
        for id, (cxt_len, ques_len) in enumerate(zip(cxt_lens, ques_lens)):
            cxt_token = cxt_corpus[cxt_count: cxt_count+cxt_len]
            cxt_token = [i for ls in cxt_token for i in ls]
            cxt_token = cxt_token[:half_seq_len]
            if len(cxt_token) < half_seq_len:
                cxt_token = cxt_token + ["<PAD>"]*(half_seq_len - len(cxt_token))

            ques_token = ques_corpus[ques_count: ques_count+ques_len]
            ques_token = [i for ls in ques_token for i in ls]
            ques_token = ques_token[:half_seq_len]
            if len(ques_token) < half_seq_len:
                ques_token = ques_token + ["<PAD>"]*(half_seq_len - len(ques_token))
            
            start_word_id = df.iloc[id]["Answer Start ID"]
            num_word = df.iloc[id]["Num Words"]
            start_id = len([i for ls in cxt_corpus[cxt_count:cxt_count+start_word_id] for i in ls])
            end_id = len([i for ls in cxt_corpus[cxt_count:cxt_count+start_word_id+num_word] for i in ls])

            if end_id <= half_seq_len:
                cxt_tokens.append([self.word2id[ch] if ch in vocab else self.word2id["<UNK>"] for ch in ["<CLS>"] + cxt_token])
                ques_tokens.append([self.word2id[ch] if ch in vocab else self.word2id["<UNK>"] for ch in ["<SEP>"] + ques_token])
                start_ids.append(start_id) 
                end_ids.append(end_id)
                topics.append(df.iloc[id]["Topic"])
            
            cxt_count += cxt_len
            ques_count += ques_len

        tokens = np.concatenate([ques_tokens, cxt_tokens], axis=-1)

        return tokens, np.array(start_ids), np.array(end_ids), np.array(topics)

    def preprocess_text(self, text):
        text = preprocess_text(text, self.operations)

        return text
    
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
        with open(self.input_path, 'r') as f:
            data = json.load(f)

        data = data["data"]
        topics, contexts, ques, ans_texts,   = [], [], [], []
        ans_start_ids, num_words, valid_ids = [], [], []

        for topic in data[:self.num_topics]:
            for para in topic["paragraphs"]:
                for que in para["qas"]:
                    topics.append(topic["title"])
                    contexts.append(para["context"])
                    ques.append(que["question"])
                    ans_texts.append(que["answers"][0]["text"])

        for j, (ans, context) in enumerate(zip(ans_texts, contexts)):
            num_word = len(ans.split())
            tokens = context.split()
            
            for i in range(len(tokens) - num_word):
                is_ans = " ".join(tokens[i:i+num_word])
                if is_ans == ans:
                    ans_start_ids.append(i)
                    num_words.append(num_word)
                    valid_ids.append(j)
                    break

        valid_ids = np.array(valid_ids)

        df = pd.DataFrame.from_dict({
            "Topic": np.array(topics)[valid_ids],
            "Context": np.array(contexts)[valid_ids],
            "Question": np.array(ques)[valid_ids],
            "Answer Start ID": ans_start_ids,
            "Num Words": num_words
        })

        return df


def create_data_loader_finetune(tokens, start_ids, end_ids, topics, val_split=0.2, test_split=0.2, batch_size=32, seed=2024):
    train_tokens, val_tokens, train_start_ids, val_start_ids, train_end_ids, val_end_ids, _, val_topics = train_test_split(tokens, start_ids, end_ids, topics, test_size=val_split+test_split, random_state=seed, stratify=topics)
    val_tokens, test_tokens, val_start_ids, test_start_ids, val_end_ids, test_end_ids, val_topics, _ = train_test_split(val_tokens, val_start_ids, val_end_ids, val_topics, test_size=test_split/(val_split+test_split), random_state=seed, stratify=val_topics)

    train_ds = TensorDataset(torch.Tensor(train_tokens), torch.Tensor(train_start_ids), torch.Tensor(train_end_ids))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

    val_ds = TensorDataset(torch.Tensor(val_tokens), torch.Tensor(val_start_ids), torch.Tensor(val_end_ids))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True)

    test_ds = TensorDataset(torch.Tensor(test_tokens), torch.Tensor(test_start_ids), torch.Tensor(test_end_ids))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1, pin_memory=False)

    return train_loader, val_loader, test_loader