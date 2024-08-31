import os
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

def plot_topk_cooccur_matrix(cooccur_mat, vocab, output_folder, k=20):
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    sns.heatmap(cooccur_mat[:k, :k], xticklabels=vocab[:k], yticklabels=vocab[:k], ax=ax)

    fig.savefig(os.path.join(output_folder, "TopK Cooccurence Matrix.png"))