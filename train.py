import json
import numpy as np

# XGBoost Regressor model load
from xgboost import XGBRegressor

from tqdm import tqdm
from utils import load
from utils import preprocessing

from utils import loss_fn
from utils import metrics
# TODO : (train_split, valid_split), (train, test) 쌍 train 훈련 저장시키기?
# TODO : best_iterations json에 저장시키기

def train(conf):
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

    # Early stopping을 이용하여 건물별 best iteration 저장 
    best_iterations_dict = dict()
    
    all_y_valid = []
    all_y_pred = []

    # 빌딩 start, end 정의
    start, end = 1, 100

    print("⚡ model train")
    # building별로 모델 학습 및 훈련하기 
    for building_num in tqdm(range(start, end + 1)):
        
        X_train = new_train_df[new_train_df["building_num"] == building_num].drop(['building_num', 'power'], axis=1)
        y_train = new_train_df[new_train_df["building_num"] == building_num]['power']

        X_valid = new_valid_df[new_valid_df["building_num"] == building_num].drop(['building_num', 'power'], axis=1)
        y_valid = new_valid_df[new_valid_df["building_num"] == building_num]['power']
        
        X_train['week'] = X_train['week'].astype('int64')
        X_valid['week'] = X_valid['week'].astype('int64')
        
        xgb_reg = XGBRegressor(n_estimators = 10000, eta = 0.01, min_child_weight = 6, 
                               max_depth = 5, colsample_bytree = 0.5, 
                               subsample = 0.9, seed=42)
        
        xgb_reg.set_params(**{'objective':loss_fn.weighted_mse(2)})
        xgb_reg.set_params(**{'early_stopping_rounds':1000})
        

        xgb_reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        
        y_pred = xgb_reg.predict(X_valid)
        
        best_iterations_dict[building_num] = xgb_reg.best_iteration
        
        all_y_valid.extend(y_valid)
        all_y_pred.extend(y_pred)

        print("-"*20)
        print(f"building_num : {building_num}")
        print('best iterations: {}'.format(xgb_reg.best_iteration))
        print('SMAPE : {}'.format(metrics.SMAPE(y_valid, y_pred)))
        print()
    
    # 총 빌딩별 예측 결과값 확인하기
    print("-"*45)
    print(f'building_num {start}-{end} SMAPE : {metrics.SMAPE(np.array(all_y_valid), np.array(all_y_pred))}')
    print("-"*45)

    # best_iterations 저장하기
    iterations_path = conf.path.best_iterations_path
    with open(iterations_path, "w") as f :
        json.dump(best_iterations_dict, f)    
