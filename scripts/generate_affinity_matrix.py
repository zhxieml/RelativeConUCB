import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from utils.data import AutoEncoder
from utils.data import extract_raw_feats

NUM_EPOCH = 200
LR = 3e-4
DEVICE = torch.device("cuda")

torch.manual_seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract affinity matrix generated from libmf.")
    parser.add_argument("-i", "--input", dest="input", type=str, help="input data folder")
    args = parser.parse_args()

    input_folder = args.input
    raw_filename = os.path.join(input_folder, "raw_feats_test.txt")
    arm_filename = os.path.join(args.input, "arm_feats.npy")
    affinity_filename = os.path.join(input_folder, "affinity.npy")

    # Matrix completion.
    with open(raw_filename, "r") as raw_file:
        rol_feats, col_feats = extract_raw_feats(raw_file)

    rol_matrix = np.vstack(list(rol_feats.values()))
    col_matrix = np.vstack(list(col_feats.values()))
    rating_matrix = rol_matrix @ col_matrix.T

    # Train the auto-encoder.
    arm_feats = np.load(arm_filename, allow_pickle=True).item()
    arm_matrix = np.vstack(list(arm_feats.values()))
    num_ratings, num_feats = rating_matrix.shape[1], arm_matrix.shape[1]

    rating_matrix = torch.Tensor(rating_matrix).to(DEVICE)
    arm_matrix = torch.Tensor(arm_matrix).to(DEVICE)

    model = AutoEncoder(num_ratings, num_feats).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    user_matrix, affinity_matrix = None, None

    for epoch_idx in range(NUM_EPOCH):
        user_matrix, affinity_matrix = model(rating_matrix, arm_matrix)
        loss = criterion(affinity_matrix, rating_matrix)

        # Backward.
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # print("[Epoch {}/{}] Loss: {:.4f}".format(epoch_idx + 1, NUM_EPOCH, loss.item()))

    np.save(affinity_filename, affinity_matrix.cpu().detach().numpy())
