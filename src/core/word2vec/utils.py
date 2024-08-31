import os
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

def plot_embed(embeds, vocab, output_folder):
    tsne = TSNE(n_components=3)
    embeds_tsne = tsne.fit_transform(embeds)

    tsne_df = pd.DataFrame.from_dict({
        "X": embeds_tsne[:, 0],
        "Y": embeds_tsne[:, 1],
        "Z": embeds_tsne[:, 2], 
        "Word": vocab
    })
    fig = px.scatter_3d(tsne_df, x="X", y="Y", z="Z", text="Word")
    fig.write_html(os.path.join(output_folder, "Word Embeddings TSNE.html"))