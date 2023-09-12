import json
import os
import time

import numpy as np
from tqdm import tqdm
# XGBoost Regressor model load
from xgboost import XGBRegressor

from utils import load, loss_fn, preprocessing_fn


def inference(conf):
    # 데이터프레임 path 정의
    dir_path = conf.path.dir_path
    train_path = conf.path.train_path
    test_path = conf.path.test_path
    submission_path = conf.path.submission_path

    # 데이터프레임 불러오기
    train_df = load.load_train_df(dir_path, train_path)
    test_df = load.load_test_df(dir_path, test_path)
    submission_df = load.load_submission_df(dir_path, submission_path)
    print("⚡ Load Data Success")

    # 데이터 전처리하기
    new_train_df = preprocessing_fn.train_pre_processing(train_df)
    # TODO : valid_pre_processing 함수 만들기
    new_test_df = preprocessing_fn.test_pre_processing(test_df, train_df=train_df)
    print("⚡ Data Preprocessing Success")

    # 예측값을 담을 리스트
    all_y_pred = np.empty(0)

    # best_iterations 로드하기
    iterations_path = conf.path.best_iterations_path
    with open(iterations_path, "r") as f:
        best_iterations_dict = json.load(f)

    # 1번~100번 건물 돌리기
    for building_num in tqdm(range(1, 101)):
        X_train = new_train_df[new_train_df["building_num"] == building_num].drop(
            ["building_num", "power"], axis=1
        )
        y_train = new_train_df[new_train_df["building_num"] == building_num]["power"]

        X_test = new_test_df[new_test_df["building_num"] == building_num].drop(
            ["building_num"], axis=1
        )

        # int64로 변경
        X_train["week"] = X_train["week"].astype("int64")
        X_test["week"] = X_test["week"].astype("int64")

        # 정확한 n_estimators는 1 iteration 추가
        n_estimators = best_iterations_dict[str(building_num)] + 1

        xgb_reg = XGBRegressor(
            n_estimators=n_estimators,
            eta=0.01,
            min_child_weight=6,
            max_depth=5,
            colsample_bytree=0.5,
            subsample=0.9,
            seed=42,
        )

        xgb_reg.set_params(**{"objective": loss_fn.weighted_mse(1)})

        xgb_reg.fit(X_train, y_train)

        # 예측
        y_pred = xgb_reg.predict(X_test)
        all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)

    submission_df["answer"] = all_y_pred

    # prediction 결과 폴더없으면 생성하기
    preds_path = "./predictions"
    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    # 현재 시간을 기준으로 예측 결과 csv 생성
    now = time.strftime("%y%m%d-%H%M%S")
    submission_df.to_csv(f"./predictions/pred_{now}.csv", index=False)
