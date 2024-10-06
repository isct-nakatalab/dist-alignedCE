import numpy as np
import time
from preprocess import preprocess_adult, learn_logistic_regression
from CE.method.Dist_aligned import DistAlighedCE
from CE.method.FACE import FACE
from CE.method.NICE import _NICE
from utils import original_make, get_candidates_bad, worst_lof
from tqdm import tqdm



def exp_adult_logistic_alighedCE():
    all_data, label, _, _, _, _ = preprocess_adult()
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
    all_data, label, _, _, _, _ = preprocess_adult()
    clf = learn_logistic_regression(all_data, label)
    face = FACE(data=all_data, clf=clf,radius=3.0, epsilon=3.0, tp=0.8,td=1.5e-24)
    alighedCE = DistAlighedCE(data=all_data, clf=clf, gamma=1.0, eta=0.5, beta=0.2, lambdas=1e-4, iterations=1000)


    num_trial = 300
    seed = 102
    face_output = []
    face_worstlof_output = []
    validity_index = []

    original_input_list, oriindex_list = original_make(get_candidates_bad(0.3, all_data, clf), num_trial, seed)

    start_time = time.time()
    for i in tqdm(range(num_trial), desc="FACE"):
        original_input = np.array(original_input_list[i])
        oriindex = np.array(oriindex_list[i])
        perturbation_vector, pathcost, path = face.compute_recourse(oriindex)
        if worst_lof(path, alighedCE.Alof, all_data) == 1:
            continue
        validity_index.append(i)
        face_output.append(perturbation_vector)
        if not path:
            for j in range(1,10):
                path.append(original_input+(perturbation_vector-original_input)*j/10)
        else:
            for j in range(1,10):
                path.append(path[0]+(perturbation_vector-path[0])*j/10)
                path.append(path[len(path)-1]+(original_input-path[len(path)-1])*j/10)
            if len(path)-19 > 0:
                for k in range(len(path)-19):
                    for j in range(1,10):
                        path.append(path[k] + (path[k+1]-path[k])*j/10)
        face_worstlof_output.append(worst_lof(path, alighedCE.Alof, all_data)[0])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

def exp_adult_logistic_NICE():
    all_data, label, Dummies_Columns, Columns, df, df_label = preprocess_adult()
    clf = learn_logistic_regression(all_data, label)
    nice = _NICE(
        data=all_data,
        clf=clf,df=df,
        df_label=df_label,
        Dummies_Columns=Dummies_Columns,
        Columns=Columns,
        cat_feat=[1,3,5,6,7,8,9,13],
        num_feat=[0,2,4,10,11,12],
        to_dum=['workclass','education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        )
    alighedCE = DistAlighedCE(data=all_data, clf=clf, gamma=1.0, eta=0.5, beta=0.2, lambdas=1e-4, iterations=1000)


    num_trial = 300
    seed = 102
    nice_output = []
    nice_worstlof_output = []

    original_input_list, oriindex_list = original_make(get_candidates_bad(0.3, all_data, clf), num_trial, seed)

    start_time = time.time()
    for i in tqdm(range(num_trial), desc="NICE"):
        original_input = np.array(original_input_list[i])
        perturbation_vector, path = nice.compute_recourse(original_input)
        nice_output.append(perturbation_vector)
        nice_worstlof_output.append(worst_lof(path, alighedCE.Alof, all_data)[0])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(elapsed_time)

if __name__ == "__main__":
    # exp_adult_logistic_alighedCE()
    # exp_adult_logistic_FACE()
    exp_adult_logistic_NICE()
