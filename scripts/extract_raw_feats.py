import argparse
import os

import numpy as np

from utils.data import extract_raw_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw features generated from libmf.")
    parser.add_argument("-i", "--input", dest="input", type=str, help="input data folder")
    args = parser.parse_args()

    input_folder = args.input
    raw_filename = os.path.join(input_folder, "raw_feats_train.txt")
    arm_feats_filename = os.path.join(input_folder, "arm_raw_feats.npy")

    with open(raw_filename, "r") as raw_file:
        _, arm_feats = extract_raw_feats(raw_file)

    np.save(arm_feats_filename, arm_feats)
