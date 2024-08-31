import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(vocab_freq, output_folder):
    wordcloud = WordCloud().generate_from_frequencies(vocab_freq)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    fig.savefig(os.path.join(output_folder, "Word Cloud.png"))


def plot_topk_freq(vocab_freq, output_folder, k=10):
    vocab, freq = np.array(list(vocab_freq.keys())), np.array(list(vocab_freq.values()))
    topk_ids = np.argsort(freq)[::-1][:k]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    y = np.arange(len(vocab[topk_ids]))
    ax.barh(y, freq[topk_ids])
    ax.set_yticks(y, labels=vocab[topk_ids])
    ax.invert_yaxis()
    fig.savefig(os.path.join(output_folder, "TopK Vocab Frequency.png"))
