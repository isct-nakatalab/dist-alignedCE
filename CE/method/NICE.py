from sklearn.preprocessing import StandardScaler
from nice import NICE
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

class _NICE:
    def __init__(
            self,
            data: np.ndarray,
            clf: LogisticRegression,
            df: pd.DataFrame,
            df_label: pd.DataFrame,
            cat_feat: list,
            num_feat: list,
            to_dum: list,
            Dummies_Columns: list,
            Columns: list,
            distance_metric: str = "HEOM",
            num_normalization: str = "minmax",
            optimization: str = "proximity",
            ):
        self.data = data
        self.clf = clf
        self.df = df
        self.df_label = df_label
        self.cat_feat = cat_feat
        self.num_feat = num_feat
        self.distance_metric = distance_metric
        self.num_normalization = num_normalization
        self.optimization = optimization
        self.scaler = StandardScaler()
        self.scaler.fit(pd.get_dummies(df, drop_first=True))
        self.Dummies_Columns = Dummies_Columns
        self.Columns = Columns
        self.to_dum = to_dum
        self.predict_fn = lambda x: self.clf_nice(x)
        self.X_train = self.df.values
        self.y_train = self.df_label.values
        self.NICE_a = NICE(
            X_train=self.X_train,
            predict_fn=self.predict_fn,
            y_train=self.y_train,
            cat_feat=self.cat_feat,
            num_feat=self.num_feat,
            distance_metric = self.distance_metric,
            num_normalization = self.num_normalization,
            optimization = self.optimization,
            justified_cf=True
            )

    def inverse_standard(self, inputs): #Convert input to its pre-standardized state
        return np.round(inputs*self.scaler.scale_ + self.scaler.mean_).astype(int)

    def to_pandas_data(self, inputs): #標準化を戻したものをダミー変数を持つ状態に変換する
        inputs = pd.DataFrame(inputs, columns=self.Dummies_Columns)
        return inputs

    def to_value(self, inputs): #データフレームを値に戻す。
        return inputs.values

    def inverse_dummies(self,df_dummies, prefix_sep='_', drop_first=True):
        def extract_categories():
            categories = {}
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    categories[col] = [x for x in self.df[col].unique() if not pd.isna(x)]
            return categories

        def reorganize_dict(reference_list, data_dict):
            for key, value_list in data_dict.items():
                not_in_reference = [item for item in value_list if not any(((item in ref) and (key in ref)) for ref in reference_list)]
                in_reference = [item for item in value_list if item not in not_in_reference]
                data_dict[key] = not_in_reference + in_reference
            return data_dict

        categories = extract_categories()
        categories = reorganize_dict(self.Dummies_Columns, categories)
        inverse_df = pd.DataFrame(index = ['inputs'])
        df_dummies = df_dummies.iloc[0:1]

        for col, cats in categories.items():
            if drop_first:
                cat_dummies = [f"{col}_{cat}" for cat in cats[1:]]
            else:
                cat_dummies = [f"{col}_{cat}" for cat in cats]
            inverse_df[col] = df_dummies[cat_dummies].idxmax(axis=1).str.replace(f'{col}_', '').values

            if pd.isna(inverse_df[col]).any():
                inverse_df[col] = cats[0]
        rerutn_inverse_df = pd.DataFrame(index = ['inputs'])
        for i in self.Columns:
            if i in inverse_df.columns:
                rerutn_inverse_df[i] = inverse_df[i]
            else:
                rerutn_inverse_df[i] = df_dummies[i].values

        return rerutn_inverse_df

    def keisan_all(self, inputs):
        N = len(inputs)
        inputs = pd.DataFrame(np.array(inputs), columns=self.Columns)
        df_tentatives = pd.concat([self.df,inputs])
        inputs = pd.get_dummies(df_tentatives, columns=self.to_dum, drop_first=True).tail(N)
        inputs = self.scaler.transform(inputs)
        inputs = np.array(inputs)
        return inputs

    def clf_nice(self,inputs): #入力に対しての予測器に合うように変換
        inputs = self.keisan_all(inputs)
        pred = self.clf.predict_proba(inputs)
        pred = np.float64(pred)
        return pred

    def compute_recourse(self, original_input):
        nice_path = []
        to_explain = self.inverse_standard(original_input)
        to_explain = self.to_pandas_data(to_explain.reshape(1,-1))
        to_explain = self.inverse_dummies(to_explain, prefix_sep='_')
        to_explain = self.to_value(to_explain)
        start_time = time.time()
        CF = self.NICE_a.explain(to_explain[0:1,:])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing time of NICE: {elapsed_time} second")
        N = len(CF)
        perturbation_vector = pd.DataFrame(np.array(CF), columns=self.Columns)
        df_tentatives = pd.concat([self.df,perturbation_vector])
        perturbation_vector = pd.get_dummies(df_tentatives,columns=self.to_dum, drop_first=True).tail(N)
        perturbation_vector = self.scaler.transform(perturbation_vector)
        perturbation_vector = np.array(perturbation_vector)
        for j in range(1,10):
            nice_path.append(original_input+(perturbation_vector[0] -original_input)*j/10)
        return perturbation_vector[0], nice_path

