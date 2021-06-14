import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import *

MIN_Y = 0.495
MAX_Y = 0.515
IS_ERROR = False
NUM_ERROR = 25
FIGSIZE = (6.0, 4.5)
FONTSIZE = 14
LEGEND_FONTSIZE = 12
FIGURE_FORMAT = "pdf"

plt.style.use("ggplot")
set_plot_rc()

def plot_rewards(results, xlabel, ylabel):
    plt.clf()
    plt.figure(figsize=FIGSIZE)
    plots = {}

    for algorithm_name, result in results.items():
        num_repeat, num_iter = result.shape
        avg_reward = np.cumsum(result, axis=1) / np.array(range(1, num_iter + 1))
        mean = np.mean(avg_reward, axis=0)
        error = None
        if IS_ERROR:
            error = np.std(avg_reward, axis=0) / np.sqrt(num_repeat)
            # error *= (np.arange(0, len(error)) % (num_iter // NUM_ERROR) == 0)

        color, linestyle, label = PATTERNS[algorithm_name]
        p = plot_result_uncertainty(mean, color, label, linestyle, error)
        # p = plt.errorbar(range(len(mean)), mean, color=color, yerr=error, label=label, linestyle=linestyle)
        plots[label] = p

    handle_labels = [(plots[label], label) for label in LABELS if label in plots]
    handles, labels = zip(*handle_labels)
    renamed_labels = labels

    plt.legend(handles, tuple(renamed_labels), fontsize=LEGEND_FONTSIZE)
    # plt.grid(axis="y", linestyle="-.")
    plt.ylim(MIN_Y, MAX_Y)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=FONTSIZE)
    ax.yaxis.set_tick_params(labelsize=FONTSIZE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="input folder name")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="output folder name")
    args = parser.parse_args()

    datasetname = args.input.strip("/").split("/")[-1]
    filenames = os.listdir(args.input)
    filenames = [filename for filename in filenames if filename.startswith("all_round_rewards_") and filename.endswith(".npy")]
    results = {}

    for filename in filenames:
        splited_algorithm_name = filename.split("_")[3:-1]
        algorithm_name = "_".join(splited_algorithm_name)
        results[algorithm_name] = np.load(os.path.join(args.input, filename))

    noshare_res = {algorithm_name: results[algorithm_name] for algorithm_name in PATTERNS_NOSHARE}
    plot_rewards(noshare_res, xlabel="Iteration", ylabel="Averaged Reward")
    plt.savefig(os.path.join(args.output, "reward_{}_noshare.{}".format(datasetname, FIGURE_FORMAT)), bbox_inches="tight", format=FIGURE_FORMAT)

    shareattribute_res = {algorithm_name: results[algorithm_name] for algorithm_name in PATTERNS_SHAREATTRIBUTE}
    plot_rewards(shareattribute_res, xlabel="Iteration", ylabel="Averaged Reward")
    plt.savefig(os.path.join(args.output, "reward_{}_shareattribute.{}".format(datasetname, FIGURE_FORMAT)), bbox_inches="tight", format=FIGURE_FORMAT)
