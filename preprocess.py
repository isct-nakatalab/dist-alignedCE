import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def preprocess_artificial():

    num0 = 150 #0の点の数
    num1 = 50
    fseed = 10

    # ラベル0の点を生成
    label_0_points = num0
    np.random.seed(seed = 501)
    x_label_0 = np.linspace(0, 4, label_0_points)+ np.random.normal(0, 0.1, label_0_points)
    np.random.seed(seed = fseed+4)
    y_label_0 = np.sin(x_label_0) + np.random.normal(0, 0.2, label_0_points)

    # ラベル1の点を生成
    label_1_points = num1
    np.random.seed(seed = fseed)
    x_label_1 = np.random.normal(0, 0.2, label_1_points)
    np.random.seed(seed = fseed + 2)
    y_label_1 = np.random.normal(0, 0.2, label_1_points)

    # x座標とy座標を合わせる
    data_label_0 = np.vstack((x_label_0, y_label_0)).T
    data_label_1 = np.vstack((x_label_1, y_label_1)).T


    target_data = data_label_1
    all_data = np.vstack((data_label_0,data_label_1))
    label = np.hstack((np.zeros(num0),np.ones(num1)))
    return target_data, all_data, label, data_label_0, data_label_1

def preprocess_adult():
    df = pd.read_csv('../data/adult_feature.csv')
    df=df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns = {'education-num' : 'edunum'})
    df = df.rename(columns = {'age' : 'aged'})
    df_tentative = df
    Columns = list(df.columns)
    df = pd.get_dummies(df, drop_first=True)
    Dummies_Columns = list(df.columns)
    df_label = pd.read_csv('../data/adult_label.csv')
    df_label[df_label['income']=='<=50K']=0
    df_label[df_label['income']=='<=50K.']=0
    df_label[df_label['income']=='>50K']=1
    df_label[df_label['income']=='>50K.']=1
    df_label = df_label['income']
    df_label = df_label.astype(int)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    all_data = np.array(standardized_data)
    label = np.array(df_label)
    return all_data, label, Dummies_Columns, Columns, df_tentative

def preprocess_bank():
    df = pd.read_csv('../data/bank_feature.csv')
    df=df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns = {'age' : 'aged'})
    df = df.rename(columns = {'day_of_week' : 'day-of-week'})
    df = df.fillna('NaN')
    df_tentative = df
    Columns = list(df.columns)
    df = pd.get_dummies(df)
    Dummies_Columns = list(df.columns)
    df_label = pd.read_csv('../data/bank_label.csv')
    df_label = pd.get_dummies(df_label, drop_first=True)
    df_label = df_label['y_yes']
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    all_data = np.array(standardized_data)
    label = np.array(df_label)
    return all_data, label, Dummies_Columns, Columns, df_tentative


def learn_logistic_regression(all_data, label):
    clf = LogisticRegression()
    clf.fit(all_data, label)
    return clf

def learn_XGBoost(all_data, label):
    xgb_params = {
    'objective' : 'binary:logistic',
    'eval_metric':'logloss',
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'tree_method':'hist',
    'subsample':0.8,
    'colsample_bytree': 0.4,
    'use_label_encoder' : False,
    'seed' : 0}
    numbers = list(range(0, len(all_data)))

    seed = 100
    random.seed(seed)

    train_index = random.sample(numbers, round(len(all_data)*(3/4)))

    test_index = [i for i in numbers if i not in train_index]

    # TRAIN DATA
    train_x = all_data[train_index]
    train_y = label[train_index]

    # VALID DATA
    valid_x = all_data[test_index]
    valid_y = label[test_index]

    # TRAIN MODEL
    clf =  XGBClassifier(**xgb_params)
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0)
    return clf
