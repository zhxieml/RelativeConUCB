import numpy as np

class Arm():
    def __init__(self, aid, fv=None, related_suparms={} ):
        self.id=aid
        self.fv=fv
        self.suparms=related_suparms

class ArmManager():
    def __init__(self):
        self.arms = {}
        self.n_arms = 0
        self.dim = 0

    def load_from_dict(self, arm_dict):
        self.arms = {}

        for arm_id, arm_feat in arm_dict.items():
            self.arms[arm_id] = Arm(arm_id, arm_feat.reshape(-1, 1))
            self.dim = arm_feat.shape[0]

        self.n_arms = len(self.arms)
        self.X = np.vstack([self.arms[arm_idx].fv.T for arm_idx in range(self.n_arms)])