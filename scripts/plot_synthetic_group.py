import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import *

MIN_Y = None
MAX_Y = None
FIGSIZE = (6.0, 4.5)
FONTSIZE = 14
LEGEND_FONTSIZE = 12
FIGURE_FORMAT = "pdf"

DATASETS = {
    "synthetic_sigma_0.0": "0.0",
    "synthetic_sigma_0.5": "0.5",
    "synthetic_sigma_1.0": "1.0",
    "synthetic_sigma_2.0": "2.0",
    "synthetic_sigma_5.0": "5.0",
}
DATASETS = {
    "synthetic_sigma_5.0": "$5.0$",
    "synthetic_sigma_2.0": "$2.0$",
    "synthetic_sigma_1.0": "$1.0$",
    "synthetic_sigma_0.5": "$0.5$",
    "synthetic_sigma_0.0": "$0.0$",
}

plt.style.use("ggplot")
set_plot_rc()

def parse_algorithm_name(algorithm_name):
    # Set the mechanisms used.
    splited_algorithm_name = algorithm_name.split("_")
    basic_algorithm, select_mechanism, update_mechanism = None, None, None

    if len(splited_algorithm_name) == 3:
        basic_algorithm, select_mechanism, update_mechanism = splited_algorithm_name
    else:
        basic_algorithm = algorithm_name

    return basic_algorithm, select_mechanism, update_mechanism

def plot_bias(results, xlabel, ylabel, min_y=None, max_y=None):
    plt.clf()

    plt.figure(figsize=FIGSIZE)
    plots = {}

    for algorithm_name, result in results.items():
        cum_regrets = {datasetname: np.mean(np.sum(result[datasetname], axis=1)) for datasetname in DATASETS}

        color, linestyle, label = PATTERNS[algorithm_name]
        p, = plt.plot([DATASETS[datasetname] for datasetname in cum_regrets], list(cum_regrets.values()), color=color, label=label, linestyle=linestyle)
        plots[label] = p

    handle_labels = [(plots[label], label) for label in LABELS if label in plots]
    handles, labels = zip(*handle_labels)
    renamed_labels = labels

    plt.legend(handles, tuple(renamed_labels), fontsize=LEGEND_FONTSIZE)
    # plt.grid(axis="y", linestyle="-.")
    plt.ylim(min_y, max_y)
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel(ylabel, fontsize=FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=FONTSIZE)
    ax.yaxis.set_tick_params(labelsize=FONTSIZE)
    # apply_font(plt.gca(), FONTSIZE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="input folder name")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="output folder name")
    args = parser.parse_args()

    overall_results = {}

    for datasetname in DATASETS:
        foldername = os.path.join(args.input, datasetname)
        filenames = os.listdir(foldername)
        filenames = [filename for filename in filenames if filename.startswith("all_round_regrets_") and filename.endswith(".npy")]
        results = {}

        for filename in filenames:
            splited_algorithm_name = filename.split("_")[3:-1]
            algorithm_name = "_".join(splited_algorithm_name)
            results[algorithm_name] = np.load(os.path.join(foldername, filename))

        overall_results[datasetname] = results


    noshare_res = {
        algorithm_name: {
            datasetname: overall_results[datasetname][algorithm_name]
            for datasetname in DATASETS
        }
        for algorithm_name in PATTERNS_NOSHARE
    }
    shareattribute_res = {
        algorithm_name: {
            datasetname: overall_results[datasetname][algorithm_name]
            for datasetname in DATASETS
        }
        for algorithm_name in PATTERNS_SHAREATTRIBUTE
    }

    plot_bias(noshare_res, xlabel=r"Individual Propotion $\beta$", ylabel="Cumulative Regret")
    plt.savefig(os.path.join(args.output, "regret_group_noshare.{}".format(FIGURE_FORMAT)), bbox_inches="tight", format=FIGURE_FORMAT)

    plot_bias(shareattribute_res, xlabel=r"Individual Propotion $\beta$", ylabel="Cumulative Regret")
    plt.savefig(os.path.join(args.output, "regret_group_shareattribute.{}".format(FIGURE_FORMAT)), bbox_inches="tight", format=FIGURE_FORMAT)
