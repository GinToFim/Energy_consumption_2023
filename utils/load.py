import pandas as pd
import os

def load_df(dir_path, df_path):
    df_path = os.path.join(dir_path, df_path)
    df = pd.read_csv(df_path)
    return df

def load_train_df(dir_path, train_path):
    train_df = load_df(dir_path, train_path)
    return train_df

def load_valid_df(dir_path, valid_path):
    valid_df = load_df(dir_path, valid_path)
    return valid_df

def load_test_df(dir_path, test_path):
    test_df = load_df(dir_path, test_path)
    return test_df



if __name__ == '__main__':
    dir_path = "./data"

    train_path = "train_split.csv"
    valid_path = "valid_split.csv"

    train_df = load_train_df(dir_path, train_path)

    print(train_df)

