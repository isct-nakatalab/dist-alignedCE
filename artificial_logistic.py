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
from preprocess import preprocess_artificial, learn_logistic_regression
from CE.method.Dist_aligned import DistAlighedCE
from CE.method.FACE import FACE
from utils import original_make, get_candidates_bad, worst_lof
from tqdm import tqdm

def main():
    target_data, all_data, label, data_label_0, data_label_1 = preprocess_artificial()
    clf = learn_logistic_regression(all_data, label)
    alighedCE = DistAlighedCE(data=all_data, clf=clf, gamma=0.1, eta=0.65, beta=0.2, lambdas=0, k_dash = 10, tp=0.5, iterations=200, activate_name = "normal")
    face = FACE(data=all_data, clf=clf, radius=0.20, epsilon=0.20, tp=0.5,td=0.00006)
    input_index = 140
    original_input = data_label_0[140].reshape(1,-1)
    perturbation_vector_face, pathcost, path = face.compute_recourse(input_index)

    pointlist = np.array(path)
    pointlists_x_face = []
    for i in range(len(pointlist)):
        pointlists_x_face.append(pointlist[i][0])
    pointlists_y_face = []
    for i in range(len(pointlist)):
        pointlists_y_face.append(pointlist[i][1])

    # プロット
    pointlists_x_graph_face = pointlists_x_face.copy()
    pointlists_x_graph_face.append(original_input[0][0])
    pointlists_x_graph_face.insert(0,perturbation_vector_face[0])

    pointlists_y_graph_face = pointlists_y_face.copy()
    pointlists_y_graph_face.append(original_input[0][1])
    pointlists_y_graph_face.insert(0,perturbation_vector_face[1])

    point_list, perturbation_vector_alighedCE = alighedCE.alighed_ce(original_input, name="artificial")

    pointlists_x_ours = []
    for i in range(len(point_list)):
        pointlists_x_ours.append(point_list[i][0][0])

    pointlists_y_ours = []
    for i in range(len(point_list)):
        pointlists_y_ours.append(point_list[i][0][1])

    pointlists_x_graph_ours = pointlists_x_ours.copy()
    pointlists_x_graph_ours.append(perturbation_vector_alighedCE[0][0])
    pointlists_x_graph_ours.insert(0,original_input[0][0])

    pointlists_y_graph_ours = pointlists_y_ours.copy()
    pointlists_y_graph_ours.append(perturbation_vector_alighedCE[0][1])
    pointlists_y_graph_ours.insert(0,original_input[0][1])

    fig, axes = plt.subplots(1,2, figsize = (14, 4.8))

    axes[0].plot(pointlists_x_graph_face, pointlists_y_graph_face,c='k')
    axes[0].scatter(data_label_0[:, 0], data_label_0[:, 1], label='sample with the undesired label', marker='o',c = '0.6', alpha=0.5,s = 25)
    axes[0].scatter(data_label_1[:, 0], data_label_1[:, 1], label='sample with the desired label', marker='o',c = 'tomato', alpha=0.5,s = 25)
    axes[0].scatter(original_input[0][0], original_input[0][1], label='input', marker='o',c='#f781bf', alpha=1,s = 50)
    axes[0].scatter(pointlists_x_face, pointlists_y_face, label='feature vector in the sequence', c='b', marker='o', alpha=0.5,s = 30)
    axes[0].scatter(perturbation_vector_face[0], perturbation_vector_face[1], label='terminal of the sequence', c='r', marker='o', alpha=1.0,s = 50)

    axes[1].plot(pointlists_x_graph_ours, pointlists_y_graph_ours,c='k')
    axes[1].scatter(data_label_0[:, 0], data_label_0[:, 1], label='sample with the undesired label', marker='o',c = '0.6', alpha=0.5,s = 25)
    axes[1].scatter(data_label_1[:, 0], data_label_1[:, 1], label='sample with the desired label', marker='o',c = 'tomato', alpha=0.5,s = 25)
    axes[1].scatter(original_input[0][0], original_input[0][1], label='input', marker='o',c='#f781bf', alpha=1,s = 50)
    axes[1].scatter(pointlists_x_ours, pointlists_y_ours, label='feature vector in the sequence', c='b', marker='o', alpha=0.5,s = 30)
    axes[1].scatter(perturbation_vector_alighedCE[0][0], perturbation_vector_alighedCE[0][1], label='terminal of the sequence', c='r', marker='o', alpha=1.0,s = 50)

    plt.rcParams["font.size"] = 17
    plt.tight_layout()
    s = fig.subplotpars
    bb=[s.left, s.top+0.02, s.right-s.left, 0.05 ]
    leg = axes[0].legend(loc=8, bbox_to_anchor=bb, ncol= 3, mode="expand", borderaxespad=0,
                    bbox_transform=fig.transFigure, fancybox=False, edgecolor="k")

    plt.show()

if __name__ == "__main__":
    main()

