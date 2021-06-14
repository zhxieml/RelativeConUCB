import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from utils.data import DriftedAutoEncoder
from utils.data import extract_raw_feats

NUM_EPOCH = 1000
LR = 3e-3
DEVICE = torch.device("cuda")

torch.manual_seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract affinity matrix generated from libmf.")
    parser.add_argument("-i", "--input", dest="input", type=str, help="input data folder")
    args = parser.parse_args()

    input_folder = args.input
    rating_filename = os.path.join(args.input, "affinity_raw.npy")
    arm_filename = os.path.join(args.input, "arm_feats.npy")
    affinity_filename = os.path.join(input_folder, "affinity.npy")

    rating_matrix = np.load(rating_filename)

    # Train the auto-encoder.
    arm_feats = np.load(arm_filename, allow_pickle=True).item()
    arm_matrix = np.vstack(list(arm_feats.values()))
    (num_users, num_ratings), num_feats = rating_matrix.shape, arm_matrix.shape[1]

    rating_matrix = torch.Tensor(rating_matrix).to(DEVICE)
    arm_matrix = torch.Tensor(arm_matrix).to(DEVICE)

    model = DriftedAutoEncoder(num_ratings, num_feats, num_users).to(DEVICE)
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
