"""preprocessing_fn.py에서의 함수들을 이용하여 실질적인 preprocessing 수행
"""

import numpy as np
import pandas as pd
import preprocessing_fn

def train_pre_processing(train_df: pd.DataFrame) -> pd.DataFrame:
    """train_df DataFrame을 입력받아 전처리한다

    Args:
        train_df (pd.DataFrame): DataFrame type의 학습 데이터셋

    Returns:
        pd.DataFrame: 전처리 후의 DataFrame type의 학습 데이터셋
    """
    # 컬럼명 영어로 수정
    train_df = preprocessing_fn.rename_columns(train_df)
    
    # 결측치 채우기
    train_df = preprocessing_fn.handling_missing_values(train_df)
    
    # 시계열 - 시간 관련 변수들 생성
    train_df = preprocessing_fn.create_time_columns(train_df)

    # 근무, 점심, 저녁 시간대 변수들 생성
    train_df = preprocessing_fn.create_working_hour(train_df)

    ## 공휴일 변수 추가
    train_df = preprocessing_fn.create_holiday(train_df)
    
    # sin & cos 변수 추가
    train_df = preprocessing_fn.create_sin_cos_hour(train_df)
    
    # 화씨 온도 변수 추가
    train_df = preprocessing_fn.create_temp_f(train_df)

    # 체감 온도 변수 추가
    train_df = preprocessing_fn.create_wind_chill_temp(train_df)
    
    # Temperature Humidity Index(THI) 변수 추가
    train_df = preprocessing_fn.create_thi(train_df)

    # Cooling Degree Hour(CDH) 변수 추가
    train_df = preprocessing_fn.create_cdh(train_df)
    
    # heat_index 추가
    train_df = preprocessing_fn.create_heat_index(train_df)
    
    #  day_hour_mean 변수 추가
    day_hour_power_mean = preprocessing_fn.create_day_hour_mean(train_df)
    train_df = pd.merge(train_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])
    
    #  day_hour_std 변수 추가
    day_hour_power_std = preprocessing_fn.create_day_hour_std(train_df)
    train_df = pd.merge(train_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])
    
    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    train_df = train_df.drop(['num_date_time', 'date', 'sunshine', 'solar', 'hour'], axis=1)
    
    return train_df

def test_pre_processing(test_df, train_df=None):
    """test_df DataFrame을 입력받아 전처리한다

    Args:
        test_df (pd.DataFrame): DataFrame type의 테스트 데이터셋

    Returns:
        pd.DataFrame: 전처리 후의 DataFrame type의 테스트 데이터셋
    """
    
    # 컬럼명 영어로 수정
    test_df = preprocessing_fn.rename_columns(test_df)
    train_df = preprocessing_fn.rename_columns(train_df)
    
    # 결측치 채우기
    test_df = preprocessing_fn.handling_missing_values(test_df)
    
    # 시계열 - 시간 관련 변수들 생성
    test_df = preprocessing_fn.create_time_columns(test_df)
    train_df = preprocessing_fn.create_time_columns(train_df)
    
    # 근무, 점심, 저녁 시간대 변수들 생성
    test_df = preprocessing_fn.create_working_hour(test_df)

    ## 공휴일 변수 추가
    test_df = preprocessing_fn.create_holiday(test_df)
    
    # sin & cos 변수 추가
    test_df = preprocessing_fn.create_sin_cos_hour(test_df)
        
    # 화씨 온도 변수 추가
    test_df = preprocessing_fn.create_temp_f(test_df)

    # 체감 온도 변수 추가
    test_df = preprocessing_fn.create_wind_chill_temp(test_df)
    
    # Temperature Humidity Index(THI) 변수 추가
    test_df = preprocessing_fn.create_thi(test_df)

    # Cooling Degree Hour(CDH) 변수 추가
    test_df = preprocessing_fn.create_cdh(test_df)
    
    # heat_index 추가
    test_df = preprocessing_fn.create_heat_index(test_df)
    
    # day_hour_mean 변수 추가
    day_hour_power_mean = preprocessing_fn.create_day_hour_mean(train_df)
    test_df = pd.merge(test_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])
    
    # day_hour_std 변수 추가
    day_hour_power_std = preprocessing_fn.create_day_hour_std(train_df)
    test_df = pd.merge(test_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])
    
    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    if 'sunshine' in test_df.columns.tolist() :
        test_df = test_df.drop(['sunshine', 'solar'], axis=1)
    
    test_df = test_df.drop(['num_date_time', 'date', 'hour'], axis=1)
    
    return test_df