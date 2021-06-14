import torch

from ._base_algorithm import AlgorithmType, BaseAlgorithm
from ._conf import linucb_para

class LinUCB(BaseAlgorithm):
    def __init__(self, dim, device, is_update_all=False):
        super().__init__(dim, device)
        self.algorithm_type = AlgorithmType.LinUCB_like
        self.is_update_all = is_update_all

        # Initialize hyperparameters.
        self.lamb = linucb_para["lambda"]
        self.alpha = linucb_para["alpha"]

        # Set placeholders.
        self.M = torch.empty((0, self.dim, self.dim), device=self.device)
        self.Minv = torch.empty((0, self.dim, self.dim), device=self.device)
        self.Y = torch.empty((0, self.dim), device=self.device)
        self.theta = torch.empty((0, self.dim), device=self.device)

    def add_user(self, user_idx):
        # Initialize item parameters.
        new_M = self.lamb * torch.eye(self.dim, device=self.device)
        new_Minv = torch.eye(self.dim, device=self.device) / self.lamb
        new_Y = torch.zeros(self.dim, device=self.device)

        # Initialize user parameters.
        new_theta = torch.zeros(self.dim, device=self.device)

        # Record the incoming user.
        self.M = torch.cat((self.M, new_M.unsqueeze(0)))
        self.Minv = torch.cat((self.Minv, new_Minv.unsqueeze(0)))
        self.Y = torch.cat((self.Y, new_Y.unsqueeze(0)))
        self.theta = torch.cat((self.theta, new_theta.unsqueeze(0)))

        self.user_idx_map[user_idx] = self.num_users
        self.num_users += 1

    def decide(self, user_idx, X, selected_arm_idxs):
        if user_idx not in self.user_idx_map:
            self.add_user(user_idx)
        user_idx = self.user_idx_map[user_idx]

        return selected_arm_idxs[torch.argmax(self._get_prob(user_idx, X)).item()]

    def update(self, user_idx, x, y):
        user_idx = self.user_idx_map[user_idx]

        if self.is_update_all:
            self._update_all(x, y)
        else:
            self._update(user_idx, x, y)

    def _get_prob(self, user_idx, X):
        theta = self.theta[user_idx]
        Minv = self.Minv[user_idx]

        X_Minv = torch.mm(X, Minv)
        var = (X_Minv * X).sum(dim=1).sqrt()
        mean = torch.mv(X, theta)

        return mean + self.alpha * var

    def _update(self, user_idx, x, y):
        self.M[user_idx] += torch.ger(x, x)
        self.Minv[user_idx], _ = self._update_Minv(self.Minv[user_idx], x)
        self.Y[user_idx] += y * x
        self.theta[user_idx] = torch.mv(self.Minv[user_idx], self.Y[user_idx])

    def _update_all(self, x, y):
        self.M += (torch.ger(x, x)).unsqueeze(0).repeat(self.num_users, 1, 1)
        self.Minv, _ = self._update_Minv_all(self.Minv, x)
        self.Y += (y * x).unsqueeze(0).repeat(self.num_users, 1)
        self.theta = torch.bmm(self.Minv, self.Y.unsqueeze(2)).squeeze(2)