import numpy as np
from preprocess import preprocess_adult, learn_logistic_regression, preprocess_bank, learn_XGBoost
from CE.method.Dist_aligned import DistAlighedCE
from CE.method.FACE import FACE
from utils import original_make, get_candidates_bad, worst_lof, Valid_Flof
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from CE.method.NICE import _NICE


@hydra.main(version_base=None, config_path="conf", config_name="simulate")
def main(cfg: DictConfig) -> None:
    if cfg.setting.experience_param.data_name == "adult":
        all_data, label, Dummies_Columns, Columns, df, df_label = preprocess_adult()
    elif cfg.setting.experience_param.data_name == "bank":
        all_data, label, Dummies_Columns, Columns, df, df_label = preprocess_bank()
    else:
        raise ValueError(f"Error: '{cfg.setting.experience_param.data_name}' is not a recognized name.")

    if cfg.setting.experience_param.model_name == "logistic":
        clf = learn_logistic_regression(all_data, label)
    elif cfg.setting.experience_param.model_name == "xgboost":
        clf = learn_XGBoost(all_data, label)
    else:
        raise ValueError(f"Error: '{cfg.setting.experience_param.data_name}' is not a recognized name.")

    all_datas = all_data[:len(all_data)//30]
    face = FACE(
        data=all_datas,
        clf=clf,radius=cfg.setting.FACE_param.radius,
        epsilon=cfg.setting.FACE_param.epsilon,
        tp=cfg.setting.FACE_param.tp,
        td=cfg.setting.FACE_param.td
        )

    nice = _NICE(
        data=all_data,
        clf=clf,df=df,
        df_label=df_label,
        Dummies_Columns=Dummies_Columns,
        Columns=Columns,
        cat_feat=cfg.setting.NICE_param.cat_feat,
        num_feat=cfg.setting.NICE_param.num_feat,
        to_dum=cfg.setting.NICE_param.to_dum
        )

    alighedCE = DistAlighedCE(
        data=all_data,
        clf=clf,
        gamma=cfg.setting.alighedCE_param.gamma,
        eta=cfg.setting.alighedCE_param.eta,
        beta=cfg.setting.alighedCE_param.beta,
        lambdas=cfg.setting.alighedCE_param.lambdas,
        iterations=cfg.setting.alighedCE_param.iteration,
        activate_name=cfg.setting.alighedCE_param.activate_name,
        )

    valid_flof = Valid_Flof(alighedCE.Flof)


    num_trial = cfg.setting.experience_param.num_trial
    seed = 102
    face_output = []
    face_worstlof_output = []
    validity_index = []
    lof_output = []
    lof_worstlof_output = []
    nice_output = []
    nice_worstlof_output = []

    original_input_list, oriindex_list = original_make(get_candidates_bad(0.3, all_datas, clf), num_trial, seed)

    for i in tqdm(range(num_trial), desc="CE"):

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

        point_list, perturbation_vector = alighedCE.alighed_ce(original_input)
        lof_output.append(perturbation_vector)
        lof_worstlof_output.append(worst_lof(point_list, alighedCE.Alof, all_data)[0])

        perturbation_vector, path = nice.compute_recourse(original_input)
        nice_output.append(perturbation_vector)
        nice_worstlof_output.append(worst_lof(path, alighedCE.Alof, all_data)[0])

    print('Flof value of FACE output:',valid_flof.lof_mean(face_output),'+-',valid_flof.lof_std(face_output))
    print('Flof value of NICE output:',valid_flof.lof_mean(nice_output),'+-',valid_flof.lof_std(nice_output))
    print('Flof value of Our method output:',valid_flof.lof_mean(lof_output),'+-',valid_flof.lof_std(lof_output))

    face_worstlof_output = np.array(face_worstlof_output)
    lof_worstlof_output = np.array(lof_worstlof_output)
    nice_worstlof_output = np.array(nice_worstlof_output)
    print('Worst Alof value of FACE output:',np.mean(face_worstlof_output),'+-',np.std(face_worstlof_output))
    print('Worst Alof value of NICE output:',np.mean(nice_worstlof_output),'+-',np.std(nice_worstlof_output))
    print('Worst Alof value of Our method output:',np.mean(lof_worstlof_output),'+-',np.std(lof_worstlof_output))


if __name__ == "__main__":
    main()
