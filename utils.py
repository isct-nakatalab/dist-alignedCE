import numpy as np
import random

def original_make(data_label,num,seed):
    random.seed(seed)
    original_input_list = []
    oriindex = random.sample(tuple(data_label),num)
    for i in oriindex:
        original_input_list.append(np.array([data_label[i]]))
    return original_input_list, oriindex

def get_candidates_bad(tp, all_data, clf):
    candidates = {}
    for x_id, x in enumerate(all_data):
        if clf.predict_proba(x.reshape(1,-1))[0][1] < tp:
            candidates[x_id] = x

    return candidates

def worst_lof(pointlist, Alof, all_data):
    worstlof = -1000000
    worstpoint = np.zeros(len(all_data[1]))
    if pointlist is None:
        return 1
    for i in pointlist:
        i_lof = -Alof.score_samples(i.reshape(1,-1))
        if worstlof < i_lof:
            worstlof = i_lof
            worstpoint = i
    return worstlof,worstpoint

class Valid_Flof:
    def __init__(
            self,
            Flof,
            ):
        self.Flof = Flof

    def lof_mean(self, lists):
        tmp = []
        for i in lists:
            if isinstance(i, int):
                continue
            else:
                tmp.append(-self.Flof.score_samples(i.reshape(1,-1)))
        return np.mean(tmp)

    def lof_std(self, lists):
        tmp = []
        for i in lists:
            if isinstance(i, int):
                continue
            else:
                tmp.append(-self.Flof.score_samples(i.reshape(1,-1)))
        return np.std(tmp)
