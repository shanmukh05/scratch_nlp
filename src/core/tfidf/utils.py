import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_pairplot(X, y, output_folder, num_pcs=6, name="TFIDF PCA Pairplot"):
    pca = PCA(n_components=num_pcs)
    pca.fit(X)
    X_pca = pca.transform(X)

    X_df = pd.DataFrame(X_pca, columns=[f"PC_{i+1}" for i in range(num_pcs)])
    X_df["Label"] = y

    fig = sns.pairplot(X_df, hue="Label")
    
    fig.savefig(os.path.join(output_folder, f"{name}.png"), bbox_inches='tight')