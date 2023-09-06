import argparse
import os
import pandas as pd
from omegaconf import OmegaConf

from utils import load


def train_valid_split(conf):
    # 데이터프레임 path 정의
    dir_path = conf.path.dir_path
    train_path = conf.path.train_path

    # 데이터프레임 불러오기
    train_df = load.load_train_df(dir_path, train_path)

    # 기존 train을 나누어서 저장시킬 path
    train_split_path = os.path.join(dir_path, "train_split.csv")
    valid_split_name = os.path.join(dir_path, "valid_split.csv")

    # 빈 데이터 프레임 만들기
    train_split_df = pd.DataFrame(columns=train_df.columns)
    valid_split_df = pd.DataFrame(columns=train_df.columns)

    # 앞에 78일 * 24시간 추가하기 (78 * 24 = 1872)
    for num in range(1, 101):
        tmp_df = train_df[train_df["건물번호"] == num].iloc[:1872]
        train_split_df = pd.concat((train_split_df, tmp_df))

    # 뒤에 8일 * 24시간 추가하기
    for num in range(1, 101):
        tmp_df = train_df[train_df["건물번호"] == num].iloc[1872:]
        valid_split_df = pd.concat((valid_split_df, tmp_df))

    # 새로 나눈 train, valid를 csv 파일로 저장시키기
    train_split_df.to_csv(train_split_path, index=False)
    valid_split_df.to_csv(valid_split_name, index=False)


if __name__ == "__main__":
    # parser 객체 생성
    parser = argparse.ArgumentParser()

    # omegaconfig 파일 이름 설정하고 실행
    parser.add_argument("--config", "-c", type=str, default="base")
    args = parser.parse_args()

    # yaml 파일 load
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    print("⚡ 실행 중인 config file:", args.config)

    print("⚡ Loading train split")
    train_valid_split(conf)
    print("⚡ Success train split")
