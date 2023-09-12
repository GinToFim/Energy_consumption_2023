import json

# Hyper-parameter Searching
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# XGBoost Regressor model load
from xgboost import XGBRegressor

from utils import load, loss_fn, metrics, preprocessing


def objectiveXGBRegressor(trial: Trial, X_train, y_train, X_valid, y_valid):
    param = {
        "tree_method": "auto",
        "eta": 0.01,
        "n_estimators": 10000,
        "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "subsample": trial.suggest_categorical("subsample", [0.8, 0.9, 1.0]),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.8, 0.9, 1.0]
        ),
        "objective": loss_fn.weighted_mse(2),
        "early_stopping_rounds": 1000,
    }

    xgb_reg = XGBRegressor(**param, random_state=42)
    xgb_reg.set_params(**{"objective": loss_fn.weighted_mse(2)})

    xgb_reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    print("best iterations: {}".format(xgb_reg.best_iteration))
    y_pred = xgb_reg.predict(X_valid)

    smape = metrics.SMAPE(y_valid, y_pred)

    return smape


def opt_search(conf):
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

    # hyper parameter searching
    optuna_params = dict()
    n_trials = 25

    start, end = 1, 100

    for building_num in range(start, end + 1):
        print(f"building_num : {building_num}")

        X_train = new_train_df[new_train_df["building_num"] == building_num].drop(
            ["building_num", "power"], axis=1
        )
        y_train = new_train_df[new_train_df["building_num"] == building_num]["power"]

        X_valid = new_valid_df[new_valid_df["building_num"] == building_num].drop(
            ["building_num", "power"], axis=1
        )
        y_valid = new_valid_df[new_valid_df["building_num"] == building_num]["power"]

        X_train["week"] = X_train["week"].astype("int64")
        X_valid["week"] = X_valid["week"].astype("int64")

        # direction : score 값을 최대 또는 최소로 하는 방향으로 지정
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # n_trials : 시도 횟수 (미 입력시 무한 반복)
        study.optimize(
            lambda trial: objectiveXGBRegressor(
                trial, X_train, y_train, X_valid, y_valid
            ),
            n_trials=n_trials,
        )

        now_params = study.best_trial.params
        now_params["tree_method"] = "auto"
        now_params["eta"] = 0.01
        now_params["early_stopping_rounds"] = 1000

        new_xgb_reg = XGBRegressor(**now_params, random_state=42)
        new_xgb_reg.set_params(**{"objective": loss_fn.weighted_mse(2)})

        new_xgb_reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        now_params["n_estimators"] = new_xgb_reg.best_iteration
        print()

        optuna_params[building_num] = now_params

    optuna_path = conf.path.optuna_path
    with open(optuna_path, "w") as f:
        json.dump(optuna_params, f)
