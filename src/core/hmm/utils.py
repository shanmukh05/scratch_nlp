import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
