import sys
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
import time
import math
from tqdm import tqdm

def get_candidates_lof(tp, all_data, clf):
    candidates = []
    for x_id, x in enumerate(all_data):
        if clf.predict_proba(x.reshape(1,-1))[0][1] >= tp:
            candidates.append(x)
    return candidates

class DistAlighedCE:
    def __init__(
            self,
            data: np.ndarray,
            clf: LogisticRegression,
            gamma: float, #weight of F-lof
            eta: float, #weight of A-lof
            beta: float, #lr
            lambdas: float, #weight of reguralization
            iterations: int,
            dist_weight :float = 0.0005, #weight of l2
            tp: float = 0.80,
            k: int = 10,
            k_dash: int = 5,
            neighbor: int = 10,
            ):
        self.data = data
        self.clf = clf
        self.tp = tp
        self.k = k
        self.k_dash = k_dash
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        self.lambdas = lambdas
        self.iterations = iterations
        self.dist_weight = dist_weight
        self.neighbor = neighbor
        self.target_data = np.array(get_candidates_lof(self.tp, self.data, self.clf))
        self.Flof = LocalOutlierFactor(n_neighbors=self.k,novelty = True)
        self.Flof.fit(self.target_data)
        self.Alof = LocalOutlierFactor(n_neighbors=k_dash,novelty = True)
        self.Alof.fit(self.data)

    def calc_cost(self, perturbation_vector: np.ndarray):
        return -self.Flof.score_samples(perturbation_vector)*self.gamma -math.exp(self.Flof.score_samples(perturbation_vector))*self.eta

    def calc_grad(self, perturbation_vector: np.ndarray):
        p_score = self.calc_cost(perturbation_vector)
        p_nei = self.Alof(perturbation_vector,self.neighbor)
        nei_X = []
        nei_lof = []
        for i in range(self.neighbor):
            nei_X.append((self.data[p_nei[0][i]] - perturbation_vector)[0])
            nei_lof.append(self.calc_cost([self.data[p_nei[0][i]]]) - p_score)
        nei_X = np.array(nei_X)
        nei_lof = np.array(nei_lof)
        return  (-np.linalg.inv(nei_X.T.dot(nei_X)+self.alpha*np.eye(len(nei_X[0]))).dot(nei_X.T).dot(nei_lof)).reshape(1,-1)

    def calc_dist_grad(self, input, perturbation_vector):
        return -(perturbation_vector-input)*self.dist_weight

    def alighed_ce(self, input):
        start_time = time.time()
        pointlist = []

        perturbation_vector = input

        for i in tqdm(range(1,self.iterations),desc="alighed_ce"):

            if i%50 == 0: #get path-point
                pointlist.append(perturbation_vector)

            grad = self.calc_dist_grad(input, perturbation_vector) + self.calc_grad(perturbation_vector)
            #updata
            perturbation_vector += self.beta*grad

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"lofの処理時間: {elapsed_time}秒")

        return pointlist,perturbation_vector



