import collections
import math
import os

import numpy as np
import torch

from env import ArmManager, SupArmManager, UserManager

BUDGET_FUNCTION = lambda t: 5 * int(math.log(t + 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(in_folder):
    # Load data.
    arm_to_suparms_filename = os.path.join(in_folder, "arm_to_suparms.npy")
    arm_feats_filename = os.path.join(in_folder, "arm_feats.npy")
    affinity_filename = os.path.join(in_folder, "affinity.npy")
    arm_to_suparms = np.load(arm_to_suparms_filename, allow_pickle=True).item()
    arm_feats = np.load(arm_feats_filename, allow_pickle=True).item()
    arm_affinity_matrix = np.load(affinity_filename)

    # Construct key-term features.
    suparm_feats = collections.defaultdict(float)
    suparm_weights = collections.defaultdict(list)
    for arm_idx, related_suparm_idxs in arm_to_suparms.items():
        for suparm_idx in related_suparm_idxs:
            weight = 1.0 / len(related_suparm_idxs)
            suparm_feats[suparm_idx] += weight * arm_feats[arm_idx]
            suparm_weights[suparm_idx].append(weight)

    suparm_feats = dict(suparm_feats)
    for suparm_idx in suparm_feats:
        suparm_feats[suparm_idx] /= np.sum(suparm_weights[suparm_idx])

    # Load arms.
    arm_manager = ArmManager()
    arm_manager.load_from_dict(arm_feats)
    X = arm_manager.X
    print("Finish loading arms: {}".format(arm_manager.n_arms))

    # Load suparms.
    super_arm_manager = SupArmManager()
    super_arm_manager.load_from_dict(suparm_feats)
    super_arm_manager.load_relation(arm_to_suparms)
    tilde_X = super_arm_manager.tilde_X
    suparm_to_arms = super_arm_manager.suparm_to_arms
    print("Finish loading suparms: {}".format(super_arm_manager.num_suparm))

    return X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms
