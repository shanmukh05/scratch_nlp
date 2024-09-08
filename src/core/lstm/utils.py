import os
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history, output_folder):
    num_plots = len(history)//2
    num_x, num_y = 3, int(np.ceil(num_plots/3))

    fig, ax = plt.subplots(num_x, num_y, figsize=(15, 5*num_y))
    keys = list(set([i.split("_")[1] for i in history.keys()]))

    for i, key in enumerate(keys):
        r, c = i//3, i%3
        x = np.arange(len(history[f"train_{key}"]))
        ax[r, c].plot(x, history[f"train_{key}"], label=f"train_{key}")
        ax[r, c].plot(x, history[f"val_{key}"], label=f"val_{key}")
        ax[r, c].set_title(key)
        ax[r, c].legend()

    fig.savefig(os.path.join(output_folder, "History.png"))