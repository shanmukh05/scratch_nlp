import re
import nltk
import logging
from collections import Counter, defaultdict
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("stopwords")


def preprocess_text(text, operations=None):
    """
    _summary_

    :param text: _description_
    :type text: _type_
    :param operations: _description_, defaults to None
    :type operations: _type_, optional
    :return: _description_
    :rtype: _type_
    """
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
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """
        self.logger = logging.getLogger(__name__)

        self.num_vocab = (
            config_dict["dataset"]["num_vocab"]
            - config_dict["dataset"]["num_extra_tokens"]
        )
        self.operations = config_dict["preprocess"]["operations"]

    def fit(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        words = self.preprocess(text_ls)
        words = self.run_merge(words)

        return words

    def transform(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        words = self.preprocess(text_ls, "test")
        vocab = list(self.vocab_freq.keys())

        for i, word in enumerate(words):
            words[i] = self.merge_chars(word, vocab)

        return words

    def merge_chars(self, word, vocab):
        """
        _summary_

        :param word: _description_
        :type word: _type_
        :param vocab: _description_
        :type vocab: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :param data: _description_, defaults to "train"
        :type data: str, optional
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param words: _description_
        :type words: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param words: _description_
        :type words: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param words: _description_
        :type words: _type_
        :return: _description_
        :rtype: _type_
        """
        self.logger.info("Merging characters to achieve desired vocabulary")

        while len(self.vocab_freq) < self.num_vocab:
            words = self.build_vocab(words)
        return words


class WordPiece:
    def __init__(self, config_dict):
        """
        _summary_

        :param config_dict: _description_
        :type config_dict: _type_
        """
        self.logger = logging.getLogger(__name__)

        self.num_vocab = (
            config_dict["dataset"]["num_vocab"]
            - config_dict["dataset"]["num_extra_tokens"]
        )
        self.operations = config_dict["preprocess"]["operations"]

    def fit(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        corpus = self.preprocess(text_ls)
        corpus = self.run_merge(corpus)

        return corpus

    def transform(self, text_ls):
        """
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :return: _description_
        :rtype: _type_
        """
        corpus = self.preprocess(text_ls, "test")
        vocab = list(self.vocab_freq.keys())

        for i, word in enumerate(corpus):
            corpus[i] = self.merge_chars(word, vocab)

        return corpus

    def merge_chars(self, word, vocab):
        """
        _summary_

        :param word: _description_
        :type word: _type_
        :param vocab: _description_
        :type vocab: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param text_ls: _description_
        :type text_ls: _type_
        :param data: _description_, defaults to "train"
        :type data: str, optional
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param pair: _description_
        :type pair: _type_
        :param pair_freq: _description_
        :type pair_freq: _type_
        :return: _description_
        :rtype: _type_
        """
        p12 = pair_freq[pair]
        p1, p2 = self.vocab_freq[pair[0]], self.vocab_freq[pair[1]]
        lkhd = p12 / (p1 * p2)

        return lkhd

    def combine(self, pair):
        """
        _summary_

        :param pair: _description_
        :type pair: _type_
        :return: _description_
        :rtype: _type_
        """
        token1, token2 = pair
        return token1 + token2[2:] if token2.startswith("##") else token1 + token2

    def build_vocab(self, corpus):
        """
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        :return: _description_
        :rtype: _type_
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
        _summary_

        :param corpus: _description_
        :type corpus: _type_
        :return: _description_
        :rtype: _type_
        """
        if len(self.vocab_freq) < self.num_vocab:
            while len(self.vocab_freq) < self.num_vocab:
                corpus = self.build_vocab(corpus)
        else:
            while len(self.vocab_freq) > self.num_vocab:
                corpus = self.build_vocab(corpus)
        return corpus
