import sys
import numpy as np
import pandas as pd
import pickle as pk
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import time
import math
from sklearn.neighbors import RadiusNeighborsTransformer
from preprocess import preprocess_adult, learn_logistic_regression
from CE.method.Dist_aligned import DistAlighedCE
from CE.method.FACE import FACE
from utils import original_make, get_candidates_bad, worst_lof
from tqdm import tqdm



def exp_adult_logistic_alighedCE():
    all_data, label, _, _, _ = preprocess_adult()
    clf = learn_logistic_regression(all_data, label)
    alighedCE = DistAlighedCE(data=all_data, clf=clf, gamma=1.0, eta=0.5, beta=0.2, lambdas=1e-4, iterations=1000)

    num_trial = 300
    seed = 102
    lof_output = []
    lof_worstlof_output = []

    original_input_list, _ = original_make(get_candidates_bad(0.3, all_data, clf), num_trial, seed)

    start_time = time.time()
    for i in tqdm(range(num_trial), desc="alighedCE"):
        original_input = np.array(original_input_list[i])
        point_list, perturbation_vector = alighedCE.alighed_ce(original_input)
        lof_output.append(perturbation_vector)
        lof_worstlof_output.append(worst_lof(point_list, alighedCE.Alof, all_data)[0])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

def exp_adult_logistic_FACE():
    all_data, label, _, _, _ = preprocess_adult()
    clf = learn_logistic_regression(all_data, label)
    alighedCE = DistAlighedCE(data=all_data, clf=clf, gamma=1.0, eta=0.5, beta=0.2, lambdas=1e-4, iterations=1000)
    face = FACE(data=all_data, radius=3.0, epsilon=3.0, tp=0.8,td=1.5e-24)

    num_trial = 300
    seed = 102
    face_output = []
    lof_worstlof_output = []

    _, oriindex = original_make(get_candidates_bad(0.3, all_data, clf), num_trial, seed)

    start_time = time.time()
    for i in tqdm(range(num_trial), desc="alighedCE"):
        perturbation_vector, pathcost, path = face.compute_recourse(oriindex)
        end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)
