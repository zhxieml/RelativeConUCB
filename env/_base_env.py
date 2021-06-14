import collections
import copy
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm

from algorithm import AlgorithmType
from ._utils import *

class BaseEnv(object):
    def __init__(self, X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms,
                 out_folder, device, arm_pool_size, budget_func,
                 is_early_register=False, num_iter=200):
        self.X = X
        self.tilde_X = tilde_X
        self.arm_to_suparms = arm_to_suparms
        self.out_folder = out_folder
        self.device = device
        self.arm_pool_size = arm_pool_size
        self.budget_func = budget_func
        self.is_early_register = is_early_register
        self.num_iter = num_iter

        # Generate intermediates.
        self.num_user = len(arm_affinity_matrix)
        self.arm_affinity_matrix = arm_affinity_matrix
        self.suparm_affinity_matrix = self.generate_suparm_affinity_matrix(arm_affinity_matrix, arm_to_suparms)
        self.W = self._generate_weight_matrix(suparm_to_arms)

        # Placeholders for run-level intermediates required by specific algorithms.
        self.suparm_differences = None
        self.tilde_X_diff = None
        self.pair_to_suparms = None
        self.most_share_pair_idxs = None

        # Placeholders for conversation-level intermediates required by specific algorithms.
        self.related_pair_idxs = None
        self.tilde_X_diff_related = None

    @staticmethod
    def generate_suparm_affinity_matrix(arm_affinity_matrix, arm_to_suparms):
        rows, cols, data = [], [], []

        for arm_idx, related_suparm_idxs in arm_to_suparms.items():
            for suparm_idx in related_suparm_idxs:
                weight = 1.0 / len(related_suparm_idxs)
                rows.append(arm_idx)
                cols.append(suparm_idx)
                data.append(weight)

        # Transform from arm affinity to suparm affinity.
        arm_suparm_matrix = spp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols)))).toarray()
        arm_suparm_matrix = normalize(arm_suparm_matrix, axis=0, norm="l1")

        return arm_affinity_matrix @ arm_suparm_matrix

    @staticmethod
    def parse_algorithm_name(algorithm_name):
        # Set the mechanisms used.
        splited_algorithm_name = algorithm_name.split("_")
        basic_algorithm, select_mechanism, update_mechanism = None, None, None

        if len(splited_algorithm_name) == 3:
            basic_algorithm, select_mechanism, update_mechanism = splited_algorithm_name
        else:
            basic_algorithm = algorithm_name

        return basic_algorithm, select_mechanism, update_mechanism

    def _generate_diff_matrix(self):
        suparm_differences = {}
        for lower_suparm_idx, higher_suparm_idx in itertools.combinations(range(len(self.tilde_X)), 2):
            suparm_differences[(lower_suparm_idx, higher_suparm_idx)] = self.tilde_X[lower_suparm_idx] - self.tilde_X[higher_suparm_idx]

        return suparm_differences, np.vstack(list(suparm_differences.values()))

    def _generate_weight_matrix(self, suparm_to_arms):
        W = np.zeros((len(self.tilde_X), len(self.X)))

        for suparm_idx, arm_weights in suparm_to_arms.items():
            for arm_idx, weight in arm_weights.items():
                W[suparm_idx, arm_idx] = weight

        return W

    def _convert_suparms_to_pair(self, lower_suparm_idx, higher_suparm_idx):
        prefix_sum = lower_suparm_idx * (2 * len(self.tilde_X) - lower_suparm_idx - 1) // 2
        offset = higher_suparm_idx - lower_suparm_idx - 1

        return prefix_sum + offset

    def _get_related_pair_idxs(self, related_suparm_idxs):
        lower_higher_suparm_idxs = torch.combinations(related_suparm_idxs)

        return self._convert_suparms_to_pair(lower_higher_suparm_idxs[:, 0], lower_higher_suparm_idxs[:, 1])

    def _generate_arm_pool(self):
        selected_arm_idxs = np.random.choice(range(len(self.X)), self.arm_pool_size, replace=False)
        selected_arm_idxs = list(selected_arm_idxs)
        X_pool = self.X[selected_arm_idxs]

        return X_pool, selected_arm_idxs

    def _get_absolute_reward(self, user_idx, arm_idx, is_suparm):
        pass

    def _get_relative_reward(self, user_idx, picked_arm_idx, duel_arm_idx, is_suparm):
        pass

    @staticmethod
    def _cal_conversation_budget(frequency_func, iter):
        return frequency_func(iter) - (0 if iter == 0 else frequency_func(iter - 1))

    @staticmethod
    def _plot_results(results, iter_idxs, out_file, xlabel, ylabel):
        plt.clf()

        for algorithm_name in results:
            plt.plot(iter_idxs, results[algorithm_name], label=algorithm_name)

        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig("{}.png".format(out_file))

    def _prepare_run_services(self, required_services):
        if "diff_matrix" in required_services:
            self.suparm_differences, self.tilde_X_diff = self._generate_diff_matrix()
            self.pair_to_suparms = list(self.suparm_differences.keys())
            tensor_memory = self.tilde_X_diff.nbytes
            device_memory_total = torch.cuda.get_device_properties(self.device).total_memory
            device_memory_allocated = torch.cuda.memory_allocated(self.device)
            self.tilde_X_diff = torch.Tensor(self.tilde_X_diff)
            if device_memory_total - device_memory_allocated > tensor_memory:
                self.tilde_X_diff = self.tilde_X_diff.to(self.device)

    def _prepare_conversation_services(self, required_services, **kwargs):
        if "related_pairs" in required_services:
            self.related_pair_idxs = self._get_related_pair_idxs(kwargs["related_suparm_idxs"])
            self.tilde_X_diff_related = self.tilde_X_diff[self.related_pair_idxs].to(self.device)

    def run_algorithms(self, template_algorithms, defined_user_sequence=None, num_repeat=1):
        # Prepare intermediates for specific algorithms.
        required_services = sum([algorithm.required_services for algorithm in template_algorithms.values()], [])
        self._prepare_run_services(required_services)

        # Allocate GPU memory.
        self.X = torch.Tensor(self.X).to(self.device)
        self.tilde_X = torch.Tensor(self.tilde_X).to(self.device)
        self.W = torch.Tensor(self.W).to(self.device)

        # Overall records.
        rewards = {}
        regrets = {}
        for algorithm_name in template_algorithms:
            rewards[algorithm_name] = []
            regrets[algorithm_name] = []

        for round_idx in range(num_repeat):
            # Reinitialize algorithm instances.
            algorithms = copy.deepcopy(template_algorithms)

            # If users are registered at the beginning, add all users for all agorithms.
            if self.is_early_register:
                for algorithm_name in algorithms:
                    for user_idx in range(self.num_user):
                        algorithms[algorithm_name].add_user(user_idx)

            # Set random seed.
            torch.manual_seed(round_idx)
            random.seed(round_idx)
            np.random.seed(round_idx)

            # Round records.
            user_times = collections.defaultdict(int)
            round_rewards = {}
            round_regrets = {}
            for algorithm_name in algorithms:
                round_rewards[algorithm_name] = []
                round_regrets[algorithm_name] = []

            # Generate the user sequence.
            if defined_user_sequence is None:
                user_sequence = random.choices(range(self.num_user), k=self.num_iter)
            else:
                user_sequence = defined_user_sequence
            assert len(user_sequence) == self.num_iter, "user_sequence doesn't match"
            progress_bar = tqdm(user_sequence)
            for user_idx in progress_bar:
                # Get the number of conversations conducted in this iteration.
                conversation_budget = self._cal_conversation_budget(self.budget_func, user_times[user_idx])
                user_times[user_idx] += 1
                progress_bar.set_description("[Round {}/{}] User {} has already come {} times (budget: {})".format(
                    round_idx + 1, num_repeat, user_idx, user_times[user_idx], conversation_budget
                ))

                # Set the arms allowed to choose in this iteration.
                X_pool, selected_arm_idxs = self._generate_arm_pool()
                related_suparm_idxs = self.W[:, selected_arm_idxs].sum(dim=1).nonzero(as_tuple=True)[0]
                tilde_X_related = self.tilde_X[related_suparm_idxs]

                for algorithm_name, algorithm in algorithms.items():
                    # Key-term-level update.
                    if algorithm.algorithm_type == AlgorithmType.ConUCB_like:
                        for _ in range(conversation_budget):
                            picked_suparm_idx = algorithm.decide_attribute(user_idx, X_pool, self.tilde_X)

                            # Receive the key-term reward.
                            picked_abs_reward = self._get_absolute_reward(user_idx, picked_suparm_idx, is_suparm=True)
                            # Key-term level update.
                            algorithm.update_attribute(user_idx, self.tilde_X[picked_suparm_idx], picked_abs_reward)
                    elif algorithm.algorithm_type == AlgorithmType.RelativeConUCB_like:
                        if conversation_budget:
                            self._prepare_conversation_services(required_services, related_suparm_idxs=related_suparm_idxs)

                        for _ in range(conversation_budget):
                            picked_suparm_idx, duel_suparm_idx = algorithm.decide_attribute_pair(
                                user_idx, X_pool, self.tilde_X, tilde_X_related, self.tilde_X_diff, self.tilde_X_diff_related, self.W,
                                selected_arm_idxs, related_suparm_idxs, self.pair_to_suparms, self.related_pair_idxs, self.most_share_pair_idxs)

                            # Receive the key-term reward.
                            relative_reward = self._get_relative_reward(user_idx, picked_suparm_idx, duel_suparm_idx, is_suparm=True)

                            # Update.
                            algorithm.update_attribute_pair(user_idx, self.tilde_X[picked_suparm_idx], self.tilde_X[duel_suparm_idx], relative_reward)

                    # Item-level update.
                    picked_arm_idx = algorithm.decide(user_idx, X_pool, selected_arm_idxs)
                    picked_reward = self._get_absolute_reward(user_idx, picked_arm_idx, is_suparm=False)
                    algorithm.update(user_idx, self.X[picked_arm_idx], picked_reward)

                    # Record.
                    picked_reward = self.arm_affinity_matrix[user_idx, picked_arm_idx]
                    picked_regret = np.max(self.arm_affinity_matrix[user_idx, selected_arm_idxs]) - picked_reward
                    round_rewards[algorithm_name].append(picked_reward)
                    round_regrets[algorithm_name].append(picked_regret)

            # Merge round records into overall ones.
            for algorithm_name in algorithms:
                rewards[algorithm_name].append(round_rewards[algorithm_name])
                regrets[algorithm_name].append(round_regrets[algorithm_name])

        # Postprocess records.
        avg_rewards = {}
        avg_regrets = {}
        cum_avg_rewards = {}
        cum_regrets = {}

        for algorithm_name in template_algorithms:
            rewards[algorithm_name] = np.array(rewards[algorithm_name])
            regrets[algorithm_name] = np.array(regrets[algorithm_name])
            avg_rewards[algorithm_name] = np.mean(rewards[algorithm_name], axis=0)
            avg_regrets[algorithm_name] = np.mean(regrets[algorithm_name], axis=0)
            cum_avg_rewards[algorithm_name] = np.cumsum(avg_rewards[algorithm_name]) / np.array(range(1, self.num_iter + 1))
            cum_regrets[algorithm_name] = np.cumsum(avg_regrets[algorithm_name])

        # Save important data.
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)
        plot_cum_avg_rewards_filename = os.path.join(self.out_folder, "cum_avg_rewards")
        plot_cum_regrets_filename = os.path.join(self.out_folder, "cum_regrets")

        for algorithm_name in template_algorithms:
            suffix = "_{}_x{}".format(algorithm_name, num_repeat)
            rewards_filename = os.path.join(self.out_folder, "all_round_rewards{}".format(suffix))
            regrets_filename = os.path.join(self.out_folder, "all_round_regrets{}".format(suffix))
            cum_avg_rewards_filename = os.path.join(self.out_folder, "avg_rewards{}".format(suffix))
            cum_regrets_filename = os.path.join(self.out_folder, "cum_regrets{}".format(suffix))
            np.save(rewards_filename, rewards[algorithm_name])
            np.save(regrets_filename, regrets[algorithm_name])
            np.save(cum_avg_rewards_filename, cum_avg_rewards[algorithm_name])
            np.save(cum_regrets_filename, cum_regrets[algorithm_name])

        self._plot_results(cum_avg_rewards, range(self.num_iter), plot_cum_avg_rewards_filename, xlabel="Iteration", ylabel="Cumulative Average Reward")
        self._plot_results(cum_regrets, range(self.num_iter), plot_cum_regrets_filename, xlabel="Iteration", ylabel="Cumulative Regrets")