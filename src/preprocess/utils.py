import re
import nltk
import logging
from collections import Counter, defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("stopwords")


def preprocess_text(text, operations=None):
    """
    Preprocesses Text

    :param text: string to preprocess
    :type text: str
    :param operations: List of operations from {'lcase', 'remalpha', 'stopwords', 'stemming'}, defaults to None
    :type operations: list, optional
    :return: Preprocessed text
    :rtype: str
    """
    if "lcase" in operations or operations is None:
        # Lowercases text
        text = text.lower()
    if "remalpha" in operations or operations is None:
        # Removes Alpha Numeric characters
        text = re.sub(r"\W+", " ", text)
    if "stopwords" in operations or operations is None:
        # Removes Stopwords
        swords = stopwords.words("english")
        text = " ".join([word for word in text.split() if word not in swords])
    if "stemming" in operations or operations is None:
        # Reducing words to their stem
        snowball = SnowballStemmer(language="english")
        text = " ".join([snowball.stem(word) for word in text.split()])
    return text


class BytePairEncoding:
    """
    Byte Pair Encoding Algorithm to convert a corpus to tokens

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.num_vocab = (
            config_dict["dataset"]["num_vocab"]
            - config_dict["dataset"]["num_extra_tokens"]
        )
        self.operations = config_dict["preprocess"]["operations"]

    def fit(self, text_ls):
        """
        Fits BPE on List of sentences and Transforms into words

        :param text_ls: List of sentences
        :type text_ls: list
        :return: List of words
        :rtype: list
        """
        words = self.preprocess(text_ls)
        words = self.run_merge(words)

        return words

    def transform(self, text_ls):
        """
        Transforms list of sentences into words

        :param text_ls: List of sentences
        :type text_ls: list
        :return: List of words
        :rtype: list
        """
        words = self.preprocess(text_ls, "test")
        vocab = list(self.vocab_freq.keys())

        for i, word in enumerate(words):
            words[i] = self.merge_chars(word, vocab)

        return words

    def merge_chars(self, word, vocab):
        """
        Merging characters in a word if it's concatenation present in vocabulary

        :param word: Word
        :type word: str
        :param vocab: Vocabulary
        :type vocab: list
        :return: new word with merged characters
        :rtype: str
        """
        merge = True
        while merge:
            tokens = word.split()
            merge_count = 0

            for j in range(len(tokens) - 1):
                pair_ = (tokens[j], tokens[j + 1])
                best_chars = re.escape(" ".join(pair_))
                replace = re.compile(r"(?<!\S)" + best_chars + r"(?!\S)")

                if "".join(pair_) in vocab:
                    word = replace.sub("".join(pair_), word)
                    merge_count += 1
                    break

            if merge_count == 0:
                merge = False
        return word

    def preprocess(self, text_ls, data="train"):
        """
        Creating words from list of sentences. Words are created by adding space between each character and adding  </w> at the end. 

        :param text_ls: List od sentences
        :type text_ls: list
        :param data: {'train', 'test'} Type of data, defaults to "train"
        :type data: str, optional
        :return: List of words from all the sentences in one list
        :rtype: list
        """
        corpus = " ".join(text_ls)
        words = corpus.split()
        words = [" ".join(list(w)) + " </w>" for w in words]

        if data == "train":
            self.vocab_freq = Counter(list(corpus))
            del self.vocab_freq[" "]
            self.vocab_freq["</w>"] = len(words)

        return words

    def get_stats(self, words):
        """
        Creates a dictionary with pair of consecutive characters as key and corresponding count in corpus as value

        :param words: List of words from the corpus
        :type words: list
        :return: Dictionary with pairs of characters and frequency
        :rtype: dict
        """
        words_freq = Counter(words)
        pair_dict = defaultdict(int)
        for word, freq in words_freq.items():
            chars = word.split()
            for i in range(len(chars) - 1):
                pair_dict[(chars[i], chars[i + 1])] += freq
        return pair_dict

    def build_vocab(self, words):
        """
        Generates Vocabulary after updation of words by merging characters

        :param words: List of words
        :type words: list
        :return: List of updated words
        :rtype: list
        """
        pair_dict = self.get_stats(words)
        best_pair = max(pair_dict, key=pair_dict.get)
        best_pair_count = pair_dict[best_pair]

        self.vocab_freq["".join(best_pair)] = best_pair_count
        self.vocab_freq[best_pair[0]] -= best_pair_count
        self.vocab_freq[best_pair[1]] -= best_pair_count

        if self.vocab_freq[best_pair[0]] == 0:
            del self.vocab_freq[best_pair[0]]
        if self.vocab_freq[best_pair[1]] == 0:
            del self.vocab_freq[best_pair[1]]

        best_chars = re.escape(" ".join(best_pair))
        replace = re.compile(r"(?<!\S)" + best_chars + r"(?!\S)")

        for i, word in enumerate(words):
            words[i] = replace.sub("".join(best_pair), word)

        return words

    def run_merge(self, words):
        """
        Updates vocabulary until desired Vocabulary count is reached

        :param words: List of words
        :type words: list
        :return: List of updated final words
        :rtype: list
        """
        self.logger.info("Merging characters to achieve desired vocabulary")

        while len(self.vocab_freq) < self.num_vocab:
            words = self.build_vocab(words)
        return words


class WordPiece:
    """
    WordPiece Tokenization Algorithm to tokenize a corpus and generate Vocabulary

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)

        self.num_vocab = (
            config_dict["dataset"]["num_vocab"]
            - config_dict["dataset"]["num_extra_tokens"]
        )
        self.operations = config_dict["preprocess"]["operations"]

    def fit(self, text_ls):
        """
        Fits WordPiece on List of sentences and Transforms the words

        :param text_ls: List of sentences
        :type text_ls: list
        :return: List of words
        :rtype: list
        """
        corpus = self.preprocess(text_ls)
        corpus = self.run_merge(corpus)

        return corpus

    def transform(self, text_ls):
        """
        Transforms list of sentences into words

        :param text_ls: List of sentences
        :type text_ls: list
        :return: List of words
        :rtype: list
        """
        corpus = self.preprocess(text_ls, "test")
        vocab = list(self.vocab_freq.keys())

        for i, word in enumerate(corpus):
            corpus[i] = self.merge_chars(word, vocab)

        return corpus

    def merge_chars(self, word, vocab):
        """
        Merging characters in a word if it's concatenation present in vocabulary

        :param word: Word
        :type word: str
        :param vocab: Vocabulary
        :type vocab: list
        :return: new word with merged characters
        :rtype: str
        """
        j = 0
        while j < len(word) - 1:
            ch1, ch2 = word[j], word[j + 1]
            new_ch = self.combine((ch1, ch2))
            if new_ch in vocab:
                word = word[:j] + [new_ch] + word[j + 2 :]
            else:
                j += 1
        return word

    def preprocess(self, text_ls, data="train"):
        """
        Creating words from list of sentences. Words are created by adding ## at start each character (other than first character). 

        :param text_ls: List od sentences
        :type text_ls: list
        :param data: {'train', 'test'} Type of data, defaults to "train"
        :type data: str, optional
        :return: List of words from all the sentences in one list. Each word is a list of characters 
        :rtype: list
        """
        words = " ".join(text_ls).split()
        corpus = []

        self.vocab_freq = defaultdict()
        for word in words:
            chars = []
            for i, ch in enumerate(word):
                if i != 0:
                    ch = f"##{ch}"
                chars.append(ch)
                if data == "train":
                    self.vocab_freq[ch] = self.vocab_freq.get(ch, 0) + 1
            corpus.append(chars)

        return corpus

    def get_stats(self, corpus):
        """
        Creates a dictionary with pair of consecutive characters as key and corresponding count in corpus as value

        :param corpus: List of words 
        :type corpus: list
        :return: Dictionary with pairs of characters and frequency
        :rtype: dict
        """
        pair_freq = defaultdict(int)
        for corp in corpus:
            if len(corp) == 1:
                continue
            for i in range(len(corp) - 1):
                pair_freq[(corp[i], corp[i + 1])] += 1
        return pair_freq

    def get_likelihood(self, pair, pair_freq):
        """
        Calculates likelihood of two characters being consecutive in a corpus

        :param pair: Pair of characters
        :type pair: tuple
        :param pair_freq: Dictionary with pairs of characters and frequency
        :type pair_freq: dict
        :return: Likelihood (can be a value in [0, 1])
        :rtype: float
        """
        p12 = pair_freq[pair]
        p1, p2 = self.vocab_freq[pair[0]], self.vocab_freq[pair[1]]
        lkhd = p12 / (p1 * p2)

        return lkhd

    def combine(self, pair):
        """
        Combines pair of characters based on their location in a word by removing ##

        :param pair: Pair of characters
        :type pair: tuple
        :return: Combination of characters
        :rtype: str
        """
        token1, token2 = pair
        return token1 + token2[2:] if token2.startswith("##") else token1 + token2

    def build_vocab(self, corpus):
        """
        Generates Vocabulary after updation of words by merging characters

        :param corpus: List of words
        :type corpus: list
        :return: List of updated corpus
        :rtype: list
        """
        pair_freq = self.get_stats(corpus)
        best_pair = max(
            pair_freq.keys(), key=lambda x: self.get_likelihood(x, pair_freq)
        )
        new_ch = self.combine(best_pair)
        best_pair_count = pair_freq[best_pair]

        for i, corp in enumerate(corpus):
            if len(corp) == 1:
                continue
            j = 0
            while j < len(corp) - 1:
                if (corp[j], corp[j + 1]) == best_pair:
                    corp = corp[:j] + [new_ch] + corp[j + 2 :]
                else:
                    j += 1
            corpus[i] = corp

        self.vocab_freq[new_ch] = best_pair_count
        self.vocab_freq[best_pair[0]] -= best_pair_count
        self.vocab_freq[best_pair[1]] -= best_pair_count

        if self.vocab_freq[best_pair[0]] == 0:
            del self.vocab_freq[best_pair[0]]
        if best_pair[0] != best_pair[1]:
            if self.vocab_freq[best_pair[1]] == 0:
                del self.vocab_freq[best_pair[1]]

        return corpus

    def run_merge(self, corpus):
        """
        Updates vocabulary until desired Vocabulary count is reached

        :param corpus: List of corpus
        :type corpus: list
        :return: List of updated final corpus
        :rtype: list
        """
        if len(self.vocab_freq) < self.num_vocab:
            while len(self.vocab_freq) < self.num_vocab:
                corpus = self.build_vocab(corpus)
        else:
            while len(self.vocab_freq) > self.num_vocab:
                corpus = self.build_vocab(corpus)
        return corpus
