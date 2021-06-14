import numpy as np

class SupArm():
    def __init__(self, suparm_id, fv, related_arms):
        self.id=suparm_id
        self.fv=fv
        self.related_arms=related_arms

class SupArmManager:
    def __init__(self):
        self.suparms = {}
        self.num_suparm = 0

    def load_from_dict(self, suparm_dict):
        self.suparms = {}

        for suparm_id, suparm_feat in suparm_dict.items():
            self.suparms[suparm_id] = SupArm(suparm_id, suparm_feat.reshape(-1, 1), [])

        self.num_suparm = len(self.suparms)
        self.tilde_X = np.vstack([self.suparms[suparm_idx].fv.T for suparm_idx in range(self.num_suparm)])

    def load_relation(self, arm_to_suparms):
        self.suparm_to_arms = {}

        for suparm_idx in self.suparms:
            self.suparms[suparm_idx].related_arms = {}
            self.suparm_to_arms[suparm_idx] = {}

        # Record the related arms and the contributions to them for each suparm.
        for arm_idx in arm_to_suparms:
            related_suparms = arm_to_suparms[arm_idx]

            for suparm_idx in related_suparms:
                weight = 1.0 / len(related_suparms)
                self.suparms[suparm_idx].related_arms[arm_idx] = weight
                self.suparm_to_arms[suparm_idx][arm_idx] = weight