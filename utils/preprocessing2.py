import numpy as np
import pandas as pd
import preprocessing as pre

def train_pre_processing(train_df):
    # 컬럼명 영어로 수정
    train_df = pre.rename_columns(train_df)
    
    # 결측치 채우기
    train_df = pre.handling_missing_values(train_df)
    
    # 시계열 - 시간 관련 변수들 생성
    train_df = pre.create_time_columns(train_df)

    # 근무, 점심, 저녁 시간대 변수들 생성
    train_df = pre.create_working_hour(train_df)

    ## 공휴일 변수 추가
    train_df = pre.create_holiday(train_df)
    
    # sin & cos 변수 추가
    train_df = pre.create_sin_cos_hour(train_df)
    
    # 화씨 온도 변수 추가
    train_df = pre.create_temp_f(train_df)

    # 체감 온도 변수 추가
    train_df = pre.create_wind_chill_temp(train_df)
    
    # Temperature Humidity Index(THI) 변수 추가
    train_df = pre.create_thi(train_df)

    # Cooling Degree Hour(CDH) 변수 추가
    train_df = pre.create_cdh(train_df)
    
    # heat_index 추가
    train_df = pre.create_heat_index(train_df)
    
    #  day_hour_mean 변수 추가
    day_hour_power_mean = pre.create_day_hour_mean(train_df)
    train_df = pd.merge(train_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])
    
    #  day_hour_std 변수 추가
    day_hour_power_std = pre.create_day_hour_std(train_df)
    train_df = pd.merge(train_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])
    
    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    train_df = train_df.drop(['num_date_time', 'date', 'sunshine', 'solar', 'hour'], axis=1)
    
    return train_df

def test_pre_processing(test_df, train_df=None):
    # 컬럼명 영어로 수정
    test_df = pre.rename_columns(test_df)
    train_df = pre.rename_columns(train_df)
    
    # 결측치 채우기
    test_df = pre.handling_missing_values(test_df)
    
    # 시계열 - 시간 관련 변수들 생성
    test_df = pre.create_time_columns(test_df)
    train_df = pre.create_time_columns(train_df)
    
    # 근무, 점심, 저녁 시간대 변수들 생성
    test_df = pre.create_working_hour(test_df)

    ## 공휴일 변수 추가
    test_df = pre.create_holiday(test_df)
    
    # sin & cos 변수 추가
    test_df = pre.create_sin_cos_hour(test_df)
        
    # 화씨 온도 변수 추가
    test_df = pre.create_temp_f(test_df)

    # 체감 온도 변수 추가
    test_df = pre.create_wind_chill_temp(test_df)
    
    # Temperature Humidity Index(THI) 변수 추가
    test_df = pre.create_thi(test_df)

    # Cooling Degree Hour(CDH) 변수 추가
    test_df = pre.create_cdh(test_df)
    
    # heat_index 추가
    test_df = pre.create_heat_index(test_df)
    
    #  day_hour_mean 변수 추가
    day_hour_power_mean = pre.create_day_hour_mean(train_df)
    test_df = pd.merge(test_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])
    
    #  day_hour_std 변수 추가
    day_hour_power_std = pre.create_day_hour_std(train_df)
    test_df = pd.merge(test_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])
    
    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    if 'sunshine' in test_df.columns.tolist() :
        test_df = test_df.drop(['sunshine', 'solar'], axis=1)
    
    test_df = test_df.drop(['num_date_time', 'date', 'hour'], axis=1)
    
    return test_df