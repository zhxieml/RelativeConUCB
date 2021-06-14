import argparse
import collections
import os
import random

import numpy as np
import numpy.linalg as LA

NUM_ARMS = 5000
NUM_SUPARMS = 500
NUM_USERS = 200
MAX_NUM_RELATED_ARMS = 10
MIN_NUM_RELATED_ARMS = 1
DIM_FEAT = 50
SIGMA_ARM = 1
SIGMA_SUPARM = 1
SIGMA_BASE_USER = 0
SIGMA_USER = 1
MIN_ABS_ARM_REWARD = 0.0
MAX_ABS_ARM_REWARD = 1.0

random.seed(12345)
np.random.seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    #################### ARM FEATURE ####################
    arm_to_suparms = collections.defaultdict(list)
    arm_feats = {}
    suparm_feats = {}

    # Generate a pseudo feature vector for each key-term.
    for suparm_idx in range(NUM_SUPARMS):
        suparm_feats[suparm_idx] = np.random.normal(0, SIGMA_SUPARM, (DIM_FEAT - 1,))
        num_related_arms = random.randint(MIN_NUM_RELATED_ARMS, MAX_NUM_RELATED_ARMS)
        related_arms = random.sample(range(NUM_ARMS), num_related_arms)

        for arm_idx in related_arms:
            arm_to_suparms[arm_idx].append(suparm_idx)

    # Draw the feature vector for each key-term from Gaussian.
    for arm_idx in range(NUM_ARMS):
        related_suparm_idxs = arm_to_suparms[arm_idx]

        mean_feat = np.zeros((DIM_FEAT - 1,))
        for suparm_idx in arm_to_suparms[arm_idx]:
            mean_feat += suparm_feats[suparm_idx] / len(related_suparm_idxs)

        arm_feats[arm_idx] = np.random.normal(mean_feat, SIGMA_ARM)
        arm_feats[arm_idx] /= LA.norm(arm_feats[arm_idx])

    for arm_idx in arm_feats:
        arm_feats[arm_idx] = np.append(arm_feats[arm_idx], 1)

    arm_matrix = np.vstack([arm_feats[arm_idx] for arm_idx in range(len(arm_feats))])

    #################### ARM FEATURE ####################

    #################### USER FEATURE ####################

    # Generate user feature vectors.
    user_feats = {}
    user_base_feats = np.random.normal(0, SIGMA_BASE_USER, (DIM_FEAT - 1,))

    for user_idx in range(NUM_USERS):
        user_reward_scale = np.random.uniform(MIN_ABS_ARM_REWARD, MAX_ABS_ARM_REWARD)
        user_feats[user_idx] = user_base_feats + np.random.normal(0, SIGMA_USER, (DIM_FEAT - 1,))
        user_feats[user_idx] /= LA.norm(user_feats[user_idx])
        user_feats[user_idx] *= user_reward_scale
        double_bias = np.random.uniform(user_reward_scale, 2 - user_reward_scale)
        user_feats[user_idx] = np.append(user_feats[user_idx], double_bias) / 2

    user_matrix = np.vstack([user_feats[user_idx] for user_idx in range(len(user_feats))])
    affinity_matrix = user_matrix @ arm_matrix.T

    # Save.
    output = os.path.join(args.output, "synthetic")
    if not os.path.exists(output):
        os.mkdir(output)
    np.save(os.path.join(output, "arm_to_suparms"), dict(arm_to_suparms))
    np.save(os.path.join(output, "arm_feats"), arm_feats)
    np.save(os.path.join(output, "affinity"), affinity_matrix)

    #################### USER FEATURE ####################
