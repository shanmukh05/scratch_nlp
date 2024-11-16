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
    _summary_

    :param vocab_freq: _description_
    :type vocab_freq: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    """
    wordcloud = WordCloud().generate_from_frequencies(vocab_freq)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    fig.savefig(os.path.join(output_folder, "Word Cloud.png"))


def plot_topk_freq(vocab_freq, output_folder, k=10):
    """
    _summary_

    :param vocab_freq: _description_
    :type vocab_freq: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    :param k: _description_, defaults to 10
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
    _summary_

    :param vocab_df: _description_
    :type vocab_df: _type_
    :param n: _description_
    :type n: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    :param k: _description_, defaults to 20
    :type k: int, optional
    :return: _description_
    :rtype: _type_
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
    _summary_

    :param X: _description_
    :type X: _type_
    :param y: _description_
    :type y: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    :param num_pcs: _description_, defaults to 6
    :type num_pcs: int, optional
    :param name: _description_, defaults to "TFIDF PCA Pairplot"
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
    _summary_

    :param metric_dict: _description_
    :type metric_dict: _type_
    :param output_folder: _description_
    :type output_folder: _type_
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
    _summary_

    :param data: _description_
    :type data: _type_
    :param output_folder: _description_
    :type output_folder: _type_
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
    _summary_

    :param trans_matrix_df: _description_
    :type trans_matrix_df: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(trans_matrix_df, ax=ax)
    fig.savefig(
        os.path.join(output_folder, "Transition Matrix.png"), bbox_inches="tight"
    )


def pca_emission_matrix(em_matrix_df, output_folder):
    """
    _summary_

    :param em_matrix_df: _description_
    :type em_matrix_df: _type_
    :param output_folder: _description_
    :type output_folder: _type_
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
    _summary_

    :param history: _description_
    :type history: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    :param name: _description_, defaults to "History"
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
    _summary_

    :param embeds: _description_
    :type embeds: _type_
    :param vocab: _description_
    :type vocab: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    :param fname: _description_, defaults to "Word Embeddings TSNE"
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
    _summary_

    :param y_true: _description_
    :type y_true: _type_
    :param y_pred: _description_
    :type y_pred: _type_
    :param classes: _description_
    :type classes: _type_
    :param output_folder: _description_
    :type output_folder: _type_
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    conf_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=classes, ax=ax
    )

    fig = conf_display.figure_
    fig.savefig(os.path.join(output_folder, "Confusion Matrix.png"))


def plot_topk_cooccur_matrix(cooccur_mat, vocab, output_folder, k=20):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    sns.heatmap(
        cooccur_mat[:k, :k], xticklabels=vocab[:k], yticklabels=vocab[:k], ax=ax
    )

    fig.savefig(os.path.join(output_folder, "TopK Cooccurence Matrix.png"))
