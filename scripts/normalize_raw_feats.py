import argparse
import os

import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import normalize

from utils.data import normalize_user_feats
from utils.data import normalize_arm_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize raw features generated from libmf.")
    parser.add_argument("-i", "--input", dest="input", type=str, help="input data folder")
    args = parser.parse_args()

    input_folder = args.input
    arm_filename = os.path.join(args.input, "arm_raw_feats.npy")
    arm_feats = np.load(arm_filename, allow_pickle=True).item()
    arm_feats = normalize_arm_feats(arm_feats)

    np.save(os.path.join(args.input, "arm_feats.npy"), arm_feats)
