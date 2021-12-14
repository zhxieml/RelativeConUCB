import numpy as np
import torch

from ._base_algorithm import AlgorithmType
from ._conucb import ConUCB

class RelativeConUCB(ConUCB):
    def __init__(self, dim, device, select_pair_mechanism, update_pair_mechanism, is_update_all=False, is_update_attribute_all=False):
        super().__init__(dim, device, is_update_all, is_update_attribute_all)
        self.algorithm_type = AlgorithmType.RelativeConUCB_like

        self.select_pair_mechanism = select_pair_mechanism
        self.update_pair_mechanism = update_pair_mechanism

        # Add flags to let the environment know what services the agent wants.
        self.required_services = self._parse_required_services(select_pair_mechanism)

    def decide_attribute_pair(self, user_idx, X_pool, tilde_X, tilde_X_related, tilde_X_diff, tilde_X_diff_related, W,
                          selected_arm_idxs, related_suparm_idxs, pair_to_suparms, related_pair_idxs, most_share_pair_idxs):
        picked_suparm_idx, duel_suparm_idx = None, None
        if user_idx not in self.user_idx_map:
            self.add_user(user_idx)

        if self.select_pair_mechanism == "best2":
            user_idx = self.user_idx_map[user_idx]
            picked_suparm_idx, duel_suparm_idx = self._decide_topk_attributes(user_idx, X_pool, tilde_X, k=2)
        elif self.select_pair_mechanism == "bestrelated2":
            user_idx = self.user_idx_map[user_idx]
            picked_suparm_idx_raw, duel_suparm_idx_raw = self._decide_topk_attributes(user_idx, X_pool, tilde_X_related, k=2)
            picked_suparm_idx, duel_suparm_idx = related_suparm_idxs[[picked_suparm_idx_raw, duel_suparm_idx_raw]]
        elif self.select_pair_mechanism == "doublebest2":
            user_idx = self.user_idx_map[user_idx]
            picked_suparm_idx, duel_suparm_idx = self._decide_double_attributes(user_idx, X_pool, tilde_X)
        elif self.select_pair_mechanism == "doublebestrelated2":
            user_idx = self.user_idx_map[user_idx]
            picked_suparm_idx_raw, duel_suparm_idx_raw = self._decide_double_attributes(user_idx, X_pool, tilde_X_related)
            picked_suparm_idx, duel_suparm_idx = related_suparm_idxs[[picked_suparm_idx_raw, duel_suparm_idx_raw]]
        # For difference-type algorithms, user_idx is redirected when calling self.decide_attribute().
        elif self.select_pair_mechanism == "bestdiff2":
            assert tilde_X_diff.is_cuda, "bestdiff2 method needs enough GPU memory."
            pair_idx = self.decide_attribute(user_idx, X_pool, tilde_X_diff)
            picked_suparm_idx, duel_suparm_idx = pair_to_suparms[pair_idx]
        elif self.select_pair_mechanism == "bestdiffrelated2":
            pair_idx = related_pair_idxs[self.decide_attribute(user_idx, X_pool, tilde_X_diff_related)]
            picked_suparm_idx, duel_suparm_idx = pair_to_suparms[pair_idx]
        elif self.select_pair_mechanism == "bestthendiff2":
            picked_suparm_idx = self.decide_attribute(user_idx, X_pool, tilde_X)
            duel_suparm_idx = self.decide_attribute(user_idx, X_pool, tilde_X - tilde_X[picked_suparm_idx])
        elif self.select_pair_mechanism == "bestthendiffrelated2":
            picked_suparm_idx_raw = self.decide_attribute(user_idx, X_pool, tilde_X_related)
            duel_suparm_idx_raw = self.decide_attribute(user_idx, X_pool, tilde_X_related - tilde_X_related[picked_suparm_idx_raw])
            picked_suparm_idx, duel_suparm_idx = related_suparm_idxs[[picked_suparm_idx_raw, duel_suparm_idx_raw]]

        return picked_suparm_idx, duel_suparm_idx

    def update_attribute_pair(self, user_idx, picked_tilde_x, duel_tilde_x, relative_reward):
        if self.update_pair_mechanism == "pos":
            if relative_reward:
                self.update_attribute(user_idx, picked_tilde_x, 1.0)
            else:
                self.update_attribute(user_idx, duel_tilde_x, 1.0)
        elif self.update_pair_mechanism == "pos&neg":
            if relative_reward:
                self.update_attribute(user_idx, picked_tilde_x, 1.0)
                self.update_attribute(user_idx, duel_tilde_x, 0.0)
            else:
                self.update_attribute(user_idx, picked_tilde_x, 0.0)
                self.update_attribute(user_idx,duel_tilde_x, 1.0)
        elif self.update_pair_mechanism == "difference":
            difference = picked_tilde_x - duel_tilde_x
            if relative_reward:
                self.update_attribute(user_idx, difference, 1.0)
            else:
                self.update_attribute(user_idx, -difference, 1.0)

    @staticmethod
    def _parse_required_services(select_mechanism):
        required_services = []

        if select_mechanism in ["bestdiff2", "bestdiffrelated2"]:
            required_services.append("diff_matrix")
        if select_mechanism in ["bestdiffrelated2"]:
            required_services.append("related_pairs")

        return required_services

    def _decide_double_attributes(self, user_idx, X, tilde_X):
        """ doublebest2. """
        # Decide the first one.
        picked_suparm_idx = torch.argmax(self._get_credit(user_idx, X, tilde_X)).item()

        # Pseudo update.
        tilde_x = tilde_X[picked_suparm_idx]
        self.tilde_M[user_idx] += torch.ger(tilde_x, tilde_x)
        self.tilde_Minv[user_idx], difference = self._update_Minv(self.tilde_Minv[user_idx], tilde_x)

        # Decide the second one.
        duel_suparm_idx = torch.argmax(self._get_credit(user_idx, X, tilde_X)).item()

        # Recover.
        self.tilde_M[user_idx] -= torch.ger(tilde_x, tilde_x)
        self.tilde_Minv[user_idx] += difference

        return picked_suparm_idx, duel_suparm_idx
