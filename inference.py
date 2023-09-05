import random
import json
import pandas as pd
import numpy as np
import os

# XGBoost Regressor model load
from xgboost import XGBRegressor

from tqdm import tqdm
from utils import load
from utils import preprocessing

from utils import loss_fn
from utils import metrics

def inference(conf):
    # 데이터프레임 path 정의
    dir_path = conf.path.dir_path
    train_path = conf.path.train_split_path
    valid_path = conf.path.valid_split_path


if __name__ == "__main__":
    pass