import json
import numpy as np
import pandas as pd

# XGBoost Regressor model load
from xgboost import XGBRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from tqdm import tqdm
from utils import load
from utils import preprocessing

from utils import loss_fn
from utils import metrics

def hyperparameter_searching(conf):
    # 데이터프레임 path 정의
    dir_path = conf.path.dir_path
    train_path = conf.path.train_split_path
    valid_path = conf.path.valid_split_path

    # 데이터프레임 불러오기
    train_df = load.load_train_df(dir_path, train_path) 
    valid_df = load.load_valid_df(dir_path, valid_path)
    print("⚡ Load Data Success")

    # 데이터 전처리하기
    new_train_df = preprocessing.train_pre_processing(train_df)
    # TODO : valid_pre_processing 함수 만들기
    new_valid_df = preprocessing.test_pre_processing(valid_df, train_df=train_df)
    print("⚡ Data Preprocessing Success")


    print("⚡ hyper-parameter searching start")
    df = pd.DataFrame(columns = ['n_estimators', 'eta', 'min_child_weight','max_depth', 'colsample_bytree', 'subsample'])
    preds = np.array([])

    grid = {'n_estimators' : [100], 'eta' : [0.01], 'min_child_weight' : np.arange(1, 8, 1),
            'max_depth' : np.arange(3,9,1) , 'colsample_bytree' :np.arange(0.8, 1.0, 0.1),
            'subsample' :np.arange(0.8, 1.0, 0.1)} # fix the n_estimators & eta(learning rate)

    outlying_building = [3,14,21,30,40,42,53,54,91,95,98]

    for i in tqdm(outlying_building):
        X_train = new_train_df[new_train_df["building_num"] == i].drop(['building_num', 'power'], axis=1)
        y_train = new_train_df[new_train_df["building_num"] == i]['power']

        X_valid = new_valid_df[new_valid_df["building_num"] == i].drop(['building_num', 'power'], axis=1)
        y_valid = new_valid_df[new_valid_df["building_num"] == i]['power']

        X_train['week'] = X_train['week'].astype('int64')
        X_valid['week'] = X_valid['week'].astype('int64')


        #pds = PredefinedSplit(np.append(-np.ones(len(x)-168), np.zeros(168)))
        gcv = GridSearchCV(estimator = XGBRegressor(tree_method = 'hist',
                                                    objective = loss_fn.weighted_mse(1)),
                        param_grid = grid, cv = 5, refit = True, verbose = True)


        gcv.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        best = gcv.best_estimator_
        params = gcv.best_params_
        print(params)
        pred = best.predict(X_valid)
        building = 'building '+str(i)
        print(building + ' || SMAPE : {}'.format(metrics.SMAPE(y_valid, pred)))