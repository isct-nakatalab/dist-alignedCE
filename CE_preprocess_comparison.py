import time
import random
from preprocess import preprocess_adult, learn_logistic_regression, preprocess_bank, learn_XGBoost
from CE.method.Dist_aligned import DistAlighedCE
from CE.method.FACE import FACE
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="simulate_preprocessing")
def main(cfg: DictConfig) -> None:
    if cfg.experience_param.data_name == "adult":
        all_data, label, _, _, _ = preprocess_adult()
    elif cfg.experience_param.data_name == "bank":
        all_data, label, _, _, _ = preprocess_bank()
    else:
        raise ValueError(f"Error: '{cfg.experience_param.data_name}' is not a recognized name.")

    if cfg.experience_param.model_name == "logistic":
        clf = learn_logistic_regression(all_data, label)
    elif cfg.experience_param.model_name == "xgboost":
        clf = learn_XGBoost(all_data, label)
    else:
        raise ValueError(f"Error: '{cfg.experience_param.data_name}' is not a recognized name.")

    num = 5
    all_all_data = all_data
    all_label = label
    numbers = list(range(0, len(all_data)))

    for i in range(num):
        print(f"trial:{i}")
        seed = 10
        random.seed(seed)

        all_index = random.sample(numbers, round(len(all_all_data)*((i+1)/5)))
        all_data = all_all_data[all_index]
        label = all_label[all_index]


        start_time = time.time()
        face = FACE(
            data=all_data,
            clf=clf,radius=cfg.FACE_param.radius,
            epsilon=cfg.FACE_param.epsilon,
            tp=cfg.FACE_param.tp,
            td=cfg.FACE_param.td
            )
        end_time = time.time()
        facetime = end_time - start_time
        print(f"Preprocessing time of FACE:{facetime}")

        start_time = time.time()
        alighedCE = DistAlighedCE(
            data=all_data,
            clf=clf,
            gamma=cfg.alighedCE_param.gamma,
            eta=cfg.alighedCE_param.eta,
            beta=cfg.alighedCE_param.beta,
            lambdas=cfg.alighedCE_param.lambdas,
            iterations=cfg.alighedCE_param.iteration
            )
        end_time = time.time()
        ourstime = end_time - start_time
        print(f"Preprocessing time of our method:{ourstime}")


if __name__ == "__main__":
    main()
