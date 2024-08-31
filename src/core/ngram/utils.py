import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pie_chart(vocab_df, n, output_folder, k=20):
    vocab_df.sort_values(by="Frequency", ascending=False, ignore_index=True, inplace=True)

    word_cols = [f"Word_{i+1}" for i in range(n)]
    vocab_df[word_cols] = vocab_df["Word"].str.split(" ", expand=True)

    tmp_df = vocab_df.iloc[:k]

    cmap = plt.get_cmap("rainbow")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, col in enumerate(word_cols):
        radius=i+2
        width=1
        frame = tmp_df.groupby(word_cols[:i+1])["Frequency"].sum()
        colors = cmap(get_color(frame))
        labels = [x[-1] if isinstance(x, tuple) else x for x in frame.index.to_numpy()]
        ax.pie(frame, labels=labels, colors=colors, radius=radius, wedgeprops=dict(width=width, edgecolor='w'), textprops=dict(size=20), labeldistance=0.8+i/60) 

    fig.savefig(os.path.join(output_folder, "TopK Pie Chart.png"), bbox_inches='tight')   


def get_color(l):
    s = 0
    res = [0]
    for i in range(len(l)-1):
        s += l.iloc[i]
        res.append(s / sum(l))
    return res