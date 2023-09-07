import argparse

import numpy as np
import torch
import random
import os
from omegaconf import OmegaConf

import train
import inference
import opt_search

# For reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # parser 객체 생성
    parser = argparse.ArgumentParser()

    # omegaconfig 파일 이름 설정하고 실행
    parser.add_argument("--config", "-c", type=str, default="base")
    parser.add_argument("--mode", "-m", required=True)
    args = parser.parse_args()

    # yaml 파일 load
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    # For reproducibility (Seed 고정)
    seed_everything(42) 
    
    print("⚡ 실행 중인 config file:", args.config)

    # [python main.py -m t]  or [python main.py -m train]
    if args.mode == "t" or args.mode == "train":
        train.train(conf)
    # [python main.py -m i]  or [python main.py -m inference]    
    elif args.mode == "i" or args.mode == "inference":
        inference.inference(conf)
    # [python main.py -m o] or [python main.py -m optuna]
    elif args.mode == "o" or args.mode == "optuna":
        opt_search.opt_search(conf)
    else:
        print("실행모드를 다시 입력해주세요.")
        print("train         : t,  train")
        print("inference     : i,  inference") 
        print("optuna search : o,  optuna")