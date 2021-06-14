import math

import numpy as np

def cal_dcg(scores):
    num_res = len(scores)
    weights = [1.0 / math.log2(rank + 1) for rank in range(1, num_res + 1)]
    dcg = np.dot(scores, weights)

    return dcg

def cal_ndcg(scores, ideal_scores):
    assert len(scores) == len(ideal_scores)
    dcg = cal_dcg(scores)
    ideal_dcg = cal_dcg(ideal_scores)

    return dcg / ideal_dcg