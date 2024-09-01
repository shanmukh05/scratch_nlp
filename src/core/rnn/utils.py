import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_history(history, output_folder):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for i, key in enumerate(["loss", "acc"]):
        x = np.arange(len(history[f"train_{key}"]))
        ax[i].plot(x, history[f"train_{key}"], label=f"train_{key}")
        ax[i].plot(x, history[f"val_{key}"], label=f"val_{key}")
        ax[i].legend()

    fig.savefig(os.path.join(output_folder, "History.png"))

def plot_conf_matrix(y_true, y_pred, classes, output_folder):
    conf_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=classes)

    fig = conf_display.figure_
    fig.savefig(os.path.join(output_folder, "Confusion Matrix.png"))