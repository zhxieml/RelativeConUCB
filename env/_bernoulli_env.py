import numpy as np

from ._base_env import BaseEnv

class BernoulliEnv(BaseEnv):
    def __init__(self, X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms,
                 out_folder, device, arm_pool_size, relative_noise,
                 budget_func, is_early_register=False, num_iter=200):
        super().__init__(X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms,
                         out_folder, device, arm_pool_size, budget_func,
                         is_early_register=is_early_register, num_iter=num_iter)
        self.relative_noise = relative_noise

    def _get_absolute_reward(self, user_idx, arm_idx, is_suparm):
        if is_suparm:
            affinity = self.suparm_affinity_matrix[user_idx, arm_idx]
        else:
            affinity = self.arm_affinity_matrix[user_idx, arm_idx]

        affinity = max(0.0, affinity)
        affinity = min(1.0, affinity)
        return np.random.binomial(1, affinity)

    def _get_relative_reward(self, user_idx, picked_arm_idx, duel_arm_idx, is_suparm):
        picked_affinity, duel_affinity = None, None

        if is_suparm:
            picked_affinity = self.suparm_affinity_matrix[user_idx, picked_arm_idx]
            duel_affinity = self.suparm_affinity_matrix[user_idx, duel_arm_idx]
        else:
            picked_affinity = self.arm_affinity_matrix[user_idx, picked_arm_idx]
            duel_affinity = self.arm_affinity_matrix[user_idx, duel_arm_idx]

        picked_affinity_noise = np.random.normal(scale=self.relative_noise)
        duel_affinity_noise = np.random.normal(scale=self.relative_noise)

        return picked_affinity + picked_affinity_noise > duel_affinity + duel_affinity_noise