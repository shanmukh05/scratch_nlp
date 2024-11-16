import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


def plot_wordcloud(vocab_freq, output_folder):
    """
    Generating Word Cloud Plot. Used in NGRAM, BOW

    :param vocab_freq: Vocabulary Frequency in the Corpus
    :type vocab_freq: dict
    :param output_folder: Path to saving Wordcloud png file
    :type output_folder: str
    """
    wordcloud = WordCloud().generate_from_frequencies(vocab_freq)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    fig.savefig(os.path.join(output_folder, "Word Cloud.png"))


def plot_topk_freq(vocab_freq, output_folder, k=10):
    """
    Histogram of Top K frequent Vocabulary in the corpus. Used in NGRAM, BOW

    :param vocab_freq: Vocabulary Frequency in the Corpus
    :type vocab_freq: dict
    :param output_folder: Path to saving Histogram png file
    :type output_folder: str
    :param k: Number of Vocab to plot, defaults to 10
    :type k: int, optional
    """
    vocab, freq = np.array(list(vocab_freq.keys())), np.array(list(vocab_freq.values()))
    topk_ids = np.argsort(freq)[::-1][:k]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    y = np.arange(len(vocab[topk_ids]))
    ax.barh(y, freq[topk_ids])
    ax.set_yticks(y, labels=vocab[topk_ids])
    ax.invert_yaxis()
    fig.savefig(os.path.join(output_folder, "TopK Vocab Frequency.png"))


def plot_ngram_pie_chart(vocab_df, n, output_folder, k=20):
    """
    Pie Chart Top K frequent Ngrams. Used in NGRAM

    :param vocab_df: DataFrame of Ngrams and their Frequency in Corpus
    :type vocab_df: pandas.DataFrame
    :param n: Number of terms in a Vocab (N of Ngram)
    :type n: int
    :param output_folder: Path to saving Pie Chart png file
    :type output_folder: str
    :param k: Number of Ngrams to plot, defaults to 20
    :type k: int, optional
    """
    vocab_df.sort_values(
        by="Frequency", ascending=False, ignore_index=True, inplace=True
    )

    word_cols = [f"Word_{i+1}" for i in range(n)]
    vocab_df[word_cols] = vocab_df["Word"].str.split(" ", expand=True)

    tmp_df = vocab_df.iloc[:k]

    def get_color(l):
        s = 0
        res = [0]
        for i in range(len(l) - 1):
            s += l.iloc[i]
            res.append(s / sum(l))
        return res

    cmap = plt.get_cmap("rainbow")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, col in enumerate(word_cols):
        radius = i + 2
        width = 1
        frame = tmp_df.groupby(word_cols[: i + 1])["Frequency"].sum()
        colors = cmap(get_color(frame))
        labels = [x[-1] if isinstance(x, tuple) else x for x in frame.index.to_numpy()]
        ax.pie(
            frame,
            labels=labels,
            colors=colors,
            radius=radius,
            wedgeprops=dict(width=width, edgecolor="w"),
            textprops=dict(size=20),
            labeldistance=0.8 + i / 60,
        )

    fig.savefig(os.path.join(output_folder, "TopK Pie Chart.png"), bbox_inches="tight")


def plot_pca_pairplot(X, y, output_folder, num_pcs=6, name="TFIDF PCA Pairplot"):
    """
    Pairplot of Features Colored by Labels. Used in TFIDF

    :param X: Feature 2D array (num_samples, num_features)
    :type X: numpy.ndarray
    :param y: Labels array, (num_samples, )
    :type y: numpy.ndarray
    :param output_folder: Path to saving pairplot png file
    :type output_folder: str
    :param num_pcs: Number of Features to Plot, defaults to 6
    :type num_pcs: int, optional
    :param name: Filename, defaults to "TFIDF PCA Pairplot"
    :type name: str, optional
    """
    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    X_pca = pca.transform(X)

    X_df = pd.DataFrame(X_pca, columns=[f"PC_{i+1}" for i in range(num_pcs)])
    X_df["Label"] = y

    fig = sns.pairplot(X_df, hue="Label")

    fig.savefig(os.path.join(output_folder, f"{name}.png"), bbox_inches="tight")


def viz_metrics(metric_dict, output_folder):
    """
    Visualizing Confusion Matrix and Classification Report Metrics. Used in HMM

    :param metric_dict: Metrics Dictionary with conf_matrix and clf_report as keys
    :type metric_dict: dict
    :param output_folder: Path to saving Metrics png file
    :type output_folder: str
    """
    fig, ax = plt.subplots(1, 2, figsize=(40, 15))

    sns.heatmap(
        metric_dict["conf_matrix"],
        ax=ax[0],
        annot=True,
        fmt=".10g",
        annot_kws={"size": 8},
    )
    ax[0].set_title("Confusion Matrix of Test Data")

    clf_df = pd.DataFrame.from_dict(metric_dict["clf_report"]).T
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    table = ax[1].table(
        cellText=clf_df.values,
        colLabels=clf_df.columns,
        rowLabels=clf_df.index,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax[1].set_title("Classification Report")

    fig.savefig(
        os.path.join(output_folder, "Test Predictions.png"), bbox_inches="tight"
    )


def plot_hist_dataset(data, output_folder):
    """
    Plotting KDE plot of Sentence Length and Histogram of POS tag of each token. Used in HMM

    :param data: Tuple of Train X, Test X, Train y, Test y
    :type data: tuple of (numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, seq_len], numpy.ndarray [num_samples, ], numpy.ndarray [num_samples, ])
    :param output_folder: Path to saving Data Analysis png file
    :type output_folder: str
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    train_x_len = [len(i) for i in data[0]]
    test_x_len = [len(i) for i in data[1]]
    train_y = list(itertools.chain.from_iterable(data[2]))
    train_y_unq, train_y_cnt = np.unique(train_y, return_counts=True)
    test_y = list(itertools.chain.from_iterable(data[3]))
    test_y_unq, test_y_cnt = np.unique(test_y, return_counts=True)

    sns.kdeplot(train_x_len, ax=ax[0], label="Train")
    sns.kdeplot(test_x_len, ax=ax[0], label="Test")
    ax[0].set_title("Sentence length")
    ax[0].legend()

    sns.barplot(x=train_y_cnt, y=train_y_unq, label="Train", ax=ax[1])
    sns.barplot(x=test_y_cnt, y=test_y_unq, label="Test", ax=ax[1])
    ax[1].set_title("POS Count [Normalized]")
    ax[1].legend()

    fig.savefig(os.path.join(output_folder, "Data Analysis.png"), bbox_inches="tight")


def plot_transition_matrix(trans_matrix_df, output_folder):
    """
    Heatmap of Transmission Matrix. Used in HMM

    :param trans_matrix_df: DataFrame of Transmission Matrix
    :type trans_matrix_df: pandas.DataFrame
    :param output_folder: Path to saving Heatmap png file
    :type output_folder: str
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(trans_matrix_df, ax=ax)
    fig.savefig(
        os.path.join(output_folder, "Transition Matrix.png"), bbox_inches="tight"
    )


def pca_emission_matrix(em_matrix_df, output_folder):
    """
    TSNE of Emission Matrix. Used in HMM

    :param em_matrix_df: DataFrame of Emission Matrix
    :type em_matrix_df: pandas.DataFrame
    :param output_folder: Path to saving Scatter plot as HTML file
    :type output_folder: str
    """
    vocab = list(em_matrix_df.columns)
    pos = list(em_matrix_df.index)

    tsne = TSNE(n_components=2, random_state=2023)
    arr_tsne = tsne.fit_transform(np.array(em_matrix_df[vocab]))

    fig = px.scatter(x=arr_tsne[:, 0], y=arr_tsne[:, 1], text=pos)
    fig.update_traces(textposition="bottom right")
    fig.write_html(os.path.join(output_folder, "Emission Matrix TSNE.html"))


def plot_history(history, output_folder, name="History"):
    """
    Training History with Loss, Metrics tracked during Training

    :param history: History Dictionary
    :type history: dict
    :param output_folder: Path to saving History png file
    :type output_folder: str
    :param name: Filename, defaults to "History"
    :type name: str, optional
    """
    num_plots = len(history) // 2
    num_x = int(np.ceil(num_plots / 3))
    if num_x == 1:
        num_x += 1
    num_y = 3

    fig, ax = plt.subplots(num_x, num_y, figsize=(18, 12 * num_y))
    keys = list(set(["_".join(i.split("_")[1:]) for i in history.keys()]))

    for i, key in enumerate(keys):
        r, c = i // 3, i % 3
        x = np.arange(len(history[f"train_{key}"]))
        ax[r, c].plot(x, history[f"train_{key}"], label=f"train_{key}")
        ax[r, c].plot(x, history[f"val_{key}"], label=f"val_{key}")
        ax[r, c].set_title(key)
        ax[r, c].legend()

    fig.savefig(os.path.join(output_folder, f"{name}.png"))


def plot_embed(embeds, vocab, output_folder, fname="Word Embeddings TSNE"):
    """
    3D TSNE of Word Embeddings from Embedding Matrix Layer

    :param embeds: Embeddings Array (num_samples, embed_dim)
    :type embeds: numpy.ndarray
    :param vocab: Vocabulary
    :type vocab: list
    :param output_folder: Path to saving Scatter plot as HTML file
    :type output_folder: str
    :param fname: Filename, defaults to "Word Embeddings TSNE"
    :type fname: str, optional
    """
    tsne = TSNE(n_components=3)
    embeds_tsne = tsne.fit_transform(embeds)

    tsne_df = pd.DataFrame.from_dict(
        {
            "X": embeds_tsne[:, 0],
            "Y": embeds_tsne[:, 1],
            "Z": embeds_tsne[:, 2],
            "Word": vocab,
        }
    )
    fig = px.scatter_3d(tsne_df, x="X", y="Y", z="Z", text="Word")
    fig.write_html(os.path.join(output_folder, f"{fname}.html"))


def plot_conf_matrix(y_true, y_pred, classes, output_folder):
    """
    Confusion Matrix of True Labels vs Prediction Labels. Used in GRU, RNN

    :param y_true: True Labels
    :type y_true: list
    :param y_pred: Prediction labels
    :type y_pred: list
    :param classes: List of classes
    :type classes: list
    :param output_folder: Path to saving Confusion Matrix png file
    :type output_folder: str
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    conf_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=classes, ax=ax
    )

    fig = conf_display.figure_
    fig.savefig(os.path.join(output_folder, "Confusion Matrix.png"))


def plot_topk_cooccur_matrix(cooccur_mat, vocab, output_folder, k=20):
    """
    Coocurence Matrix of Tokens in GloVe Model. Used in GLOVE

    :param cooccur_mat: CoOccurence Matrix
    :type cooccur_mat: numpy.ndarray
    :param vocab: List of Vocabulary
    :type vocab: list
    :param output_folder: Path to saving Coocurence matrix png file
    :type output_folder: str
    :param k: Number of Vocab to plot, defaults to 20
    :type k: int, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    sns.heatmap(
        cooccur_mat[:k, :k], xticklabels=vocab[:k], yticklabels=vocab[:k], ax=ax
    )

    fig.savefig(os.path.join(output_folder, "TopK Cooccurence Matrix.png"))
