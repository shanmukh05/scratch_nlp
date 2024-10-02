import re
import nltk
import logging
from collections import Counter, defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download("stopwords")


def preprocess_text(text, operations):
    if "lcase" in operations or operations is None:
        text = text.lower()
    if "remalpha" in operations or operations is None:
        text = re.sub(r"\W+", " ", text) 
    if "stopwords" in operations or operations is None:
        swords = stopwords.words("english")
        text = " ".join([word for word in text.split() if word not in swords])
    if "stemming" in operations or operations is None:
        snowball = SnowballStemmer(language="english")
        text = " ".join([snowball.stem(word) for word in text.split()])
    return text


class BytePairEncoding:
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.num_vocab = config_dict["dataset"]["num_vocab"] - 1
        self.operations = config_dict["preprocess"]["operations"]

    def fit(self, text_ls):
        words = self.preprocess(text_ls)
        words = self.run_merge(words)

        return words

    def transform(self, text_ls):
        words = self.preprocess(text_ls, "test")
        vocab = list(self.vocab_freq.keys())

        for i, word in enumerate(words):
            words[i] = self.merge_chars(word, vocab)

        return words

    def merge_chars(self, word, vocab):
        merge = True
        while merge:
            tokens = word.split()
            merge_count = 0

            for j in range(len(tokens)-1):
                pair_ = (tokens[j], tokens[j+1])
                best_chars = re.escape(" ".join(pair_))
                replace = re.compile(r'(?<!\S)' + best_chars + r'(?!\S)')

                if "".join(pair_) in vocab:
                    word = replace.sub("".join(pair_), word)
                    merge_count += 1
                    break
            
            if merge_count == 0:
                merge=False
        return word

    def preprocess(self, text_ls, data="train"):
        corpus = " ".join(text_ls)
        words = corpus.split()
        words = [" ".join(list(w))+ " </w>" for w in words]
        
        if data == "train":
            self.vocab_freq = Counter(list(corpus))
            del self.vocab_freq[" "]
            self.vocab_freq["</w>"] = len(words)

        return words

    def get_stats(self, words):
        words_freq = Counter(words)
        pair_dict = defaultdict(int)
        for word, freq in words_freq.items():
            chars = word.split()
            for i in range(len(chars)-1):
                pair_dict[(chars[i], chars[i+1])] += freq
        return pair_dict
    
    def build_vocab(self, words):
        pair_dict = self.get_stats(words)
        best_pair = max(pair_dict, key=pair_dict.get)
        best_pair_count = pair_dict[best_pair]
        
        self.vocab_freq["".join(best_pair)] = best_pair_count
        self.vocab_freq[best_pair[0]] -= best_pair_count
        self.vocab_freq[best_pair[1]] -= best_pair_count

        if self.vocab_freq[best_pair[0]] == 0: del self.vocab_freq[best_pair[0]]
        if self.vocab_freq[best_pair[1]] == 0: del self.vocab_freq[best_pair[1]]
        
        best_chars = re.escape(" ".join(best_pair))
        replace = re.compile(r'(?<!\S)' + best_chars + r'(?!\S)')

        for i, word in enumerate(words):
            words[i] = replace.sub("".join(best_pair), word)
        
        return words
    
    def run_merge(self, words):
        self.logger.info("Merging characters to achieve desired vocabulary")
        num_merges = self.num_vocab - len(self.vocab_freq)
        for _ in range(num_merges):
            words = self.build_vocab(words)
        return words