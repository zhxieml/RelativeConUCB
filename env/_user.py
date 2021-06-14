import numpy as np

class User():
    def __init__(self, uid, theta, pos_review=None, neg_review=None):
        self.uid=uid
        self.theta=theta
        self.pos_review=pos_review
        self.neg_review=neg_review

class UserManager():
    def __init__(self):
        self.users = {}
        self.n_user = 0

    def load_from_dict(self, user_dict):
        self.users = {}

        for user_id, user_feat in user_dict.items():
            self.users[user_id] = User(user_id, user_feat.reshape(-1, 1))

        self.n_user = len(self.users)
        self.U = np.vstack([self.users[user_idx].theta.T for user_idx in range(self.n_user)])