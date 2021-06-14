from enum import Enum

import torch

class AlgorithmType(Enum):
    LinUCB_like = 0
    ConUCB_like = 1
    RelativeConUCB_like = 2

class BaseAlgorithm():
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device

        # Record users.
        self.num_users = 0
        # Map the external user_idxs to internal ones.
        self.user_idx_map = {}
        # Add flags to let the environment know what services the agent wants.
        self.required_services = []

    @staticmethod
    def _update_Minv(Minv, x):
        # Update the inverse by Sherman–Morrison formula.
        Minv_x = torch.mv(Minv, x)
        result_a = torch.ger(Minv_x, Minv_x)
        result_b = 1 + (x * Minv_x).sum()
        difference = result_a / result_b

        return Minv - difference, difference

    @staticmethod
    def _update_Minv_all(Minv, x):
        assert len(Minv.shape) == 3

        Minv_x = torch.matmul(Minv, x)
        result_a = torch.bmm(Minv_x.unsqueeze(2), Minv_x.unsqueeze(1))
        result_b = 1 + (x.unsqueeze(0).repeat(len(Minv), 1) * Minv_x).sum(dim=1)
        result_b = result_b.unsqueeze(1).unsqueeze(2).repeat(1, len(x), len(x))
        difference = result_a / result_b

        return Minv - difference, difference