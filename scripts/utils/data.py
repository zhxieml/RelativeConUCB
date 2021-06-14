import numpy as np
import numpy.linalg as LA
import scipy.sparse as spp
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, num_ratings, num_feats):
        super(AutoEncoder, self).__init__()

        self._num_ratings = num_ratings
        self._num_feats = num_feats

        self._encoder = nn.Sequential(
            nn.Linear(num_ratings, num_ratings // 2),
            nn.ReLU(),
            nn.Linear(num_ratings // 2, num_ratings // 4),
            nn.ReLU(),
            nn.Linear(num_ratings // 4, num_ratings // 8),
            nn.ReLU(),
            nn.Linear(num_ratings // 8, num_ratings // 16),
            nn.ReLU(),
            nn.Linear(num_ratings // 16, num_feats - 1)
        )

    def encoder(self, rating_matrix):
        user_matrix = self._encoder(rating_matrix)
        user_matrix = F.normalize(user_matrix, p=2, dim=-1)
        user_matrix = torch.cat((user_matrix, torch.ones(len(user_matrix), 1, device=user_matrix.device)), dim=-1) / 2

        return user_matrix

    def forward(self, rating_matrix, arm_matrix):
        user_matrix = self.encoder(rating_matrix)
        affinity_matrix = user_matrix @ arm_matrix.T

        return user_matrix, affinity_matrix

class DriftedAutoEncoder(nn.Module):
    def __init__(self, num_ratings, num_feats, num_users):
        super(DriftedAutoEncoder, self).__init__()

        self._num_ratings = num_ratings
        self._num_feats = num_feats
        self._num_users = num_users
        self._embedd = nn.Embedding(1, num_feats).to()

        self._encoder = nn.Sequential(
            nn.Linear(num_ratings, num_ratings // 2),
            nn.ReLU(),
            nn.Linear(num_ratings // 2, num_ratings // 4),
            nn.ReLU(),
            nn.Linear(num_ratings // 4, num_ratings // 8),
            nn.ReLU(),
            nn.Linear(num_ratings // 8, num_ratings // 16),
            nn.ReLU(),
            nn.Linear(num_ratings // 16, 1)
        )

    def encoder(self, rating_matrix):
        user_bias_scale = self._encoder(rating_matrix)

        embedd = self._embedd(torch.LongTensor([0]).to(rating_matrix.device))[0]
        user_base_feat = embedd[1:]
        user_reward_scale = embedd[0]

        # Normalize the base feature.
        user_base_feat = F.normalize(user_base_feat, p=2, dim=-1)

        # Bound the scales into [0, 1].
        user_reward_scale = torch.sigmoid(user_reward_scale)
        user_bias_scale = torch.sigmoid(user_bias_scale)
        double_bias = user_reward_scale + (2 - 2 * user_reward_scale) * user_bias_scale

        user_matrix = user_reward_scale * user_base_feat.repeat(self._num_users, 1)
        user_matrix = torch.cat((user_matrix, double_bias), dim=1) / 2

        return user_matrix

    def forward(self, rating_matrix, arm_matrix):
        user_matrix = self.encoder(rating_matrix)
        affinity_matrix = user_matrix @ arm_matrix.T

        return user_matrix, affinity_matrix

def cal_rmse(P, Q, samples):
    error = [value - np.dot(P[row, :], Q[col, :]) for row, col, value in samples]

    return (np.dot(error, error) / len(error)) ** 0.5

def cal_rmse_crop(P, Q, samples):
    error = [value - min(max(np.dot(P[row, :], Q[col, :]), 0.0), 1.0) for row, col, value in samples]

    return (np.dot(error, error) / len(error)) ** 0.5

def extract_feats_by_svd(rating_matrix, dim):
    u, _, vt = LA.svd(rating_matrix)
    v = vt.T

    user_feats = u[:, :dim]
    item_feats = v[:, :dim]

    return user_feats, item_feats

def normalize_user_feats(user_feats):
    user_matrix = np.vstack(list(user_feats.values()))
    user_matrix_normed = normalize(user_matrix, axis=1, norm="l2")
    user_matrix_normed = np.concatenate((user_matrix_normed, np.ones((user_matrix_normed.shape[0], 1))), axis=1) / 2

    return dict(enumerate(user_matrix_normed))

def normalize_arm_feats(arm_feats):
    arm_matrix = np.vstack(list(arm_feats.values()))
    arm_matrix_normed = normalize(arm_matrix, axis=1, norm="l2")
    arm_matrix_normed = np.concatenate((arm_matrix_normed, np.ones((arm_matrix_normed.shape[0], 1))), axis=1)

    return dict(enumerate(arm_matrix_normed))

def normalize_relations(item_to_tags):
    tag_indices = {}
    item_to_tags_normalized = {}

    for tags in item_to_tags.values():
        for tag in tags:
            if tag not in tag_indices:
                tag_indices[tag] = len(tag_indices)

    for item, related_tags in item_to_tags.items():
        item_to_tags_normalized[item] = [tag_indices[tag] for tag in related_tags]

    return item_to_tags_normalized

def extract_rows(top_k, sparse_matrix):
    user_rating_count = sparse_matrix.getnnz(axis=1)
    user_count = user_rating_count.shape[0]

    top_k_indices = np.argsort(user_rating_count)[-1 : user_count - 1 - top_k : -1]
    matrix = spp.vstack([sparse_matrix.getrow(i) for i in top_k_indices])

    return top_k_indices, matrix

def extract_cols(top_k, sparse_matrix):
    item_rating_count = sparse_matrix.getnnz(axis=0)
    item_count = item_rating_count.shape[0]

    top_k_indices = np.argsort(item_rating_count)[-1 : item_count - 1 - top_k :-1]
    matrix = spp.hstack([sparse_matrix.getcol(i) for i in top_k_indices])

    return top_k_indices, matrix

def get_reduced_concrete_matrix(full_matrix, num_user, num_item):
    _, row_reduced_matrix = extract_rows(num_user * 3, full_matrix)
    col_indices, reduced_matrix = extract_cols(num_item, row_reduced_matrix)
    _, reduced_matrix = extract_rows(num_user, reduced_matrix)

    return reduced_matrix.toarray(), col_indices

def save_relations(output, item_to_tags, delimiter=","):
    with open(output, "w") as f:
        for idx, item in enumerate(item_to_tags):
            related_tags = item_to_tags[item]
            f.write("{}\t{}\n".format(idx, delimiter.join([str(tag) for tag in related_tags])))

def save_oneclass(output, rating_matrix):
    with open(output, "w") as f:
        for row, col in zip(*np.nonzero(rating_matrix)):
            f.write("{} {} 1\n".format(row, col))

def extract_raw_feats(raw_file):
    user_feats, arm_feats = {}, {}

    for line in raw_file.readlines():
        # User features.
        if line[0] == "p":
            feat = np.array(line.strip().split(" ")[2:]).astype(float)
            user_feats[len(user_feats)] = np.array(feat)
        # Arm features.
        elif line[0] == "q":
            feat = np.array(line.strip().split(" ")[2:]).astype(float)
            arm_feats[len(arm_feats)] = np.array(feat)

    return user_feats, arm_feats
