#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# data pre-processing load
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

# others ML model load
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from tqdm import tqdm


# In[2]:


# For reproducibility

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42) # seed 고정


# In[ ]:


# path 설정

dir_path = "./data/energy_cosumption"

train_path = os.path.join(dir_path, "train_split.csv")
original_train_path = os.path.join(dir_path, "train.csv")

valid_path = os.path.join(dir_path, "valid_split.csv")

test_path = os.path.join(dir_path, "test.csv")

building_path = os.path.join(dir_path, "building_info.csv")
submission_path = os.path.join(dir_path, "sample_submission.csv") 


# In[ ]:


# Load 데이터프레임

train_df = pd.read_csv(train_path)
original_train_df = pd.read_csv(original_train_path)

valid_df = pd.read_csv(valid_path)

test_df = pd.read_csv(test_path)

building_df = pd.read_csv(building_path)
submission_df = pd.read_csv(submission_path)


# In[ ]:


# 빌딩 별 전력사용량 평균
def mean_of_power_per_building(df):
    train_df = pd.read_csv('./train.csv')
    train_df = train_df.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    })
    train_df = train_df.fillna(0)
    train_df.drop('num_date_time', axis = 1, inplace=True)

    train_df['date_time'] = pd.to_datetime(train_df['date_time'])
    train_df['year'] = train_df['date_time'].dt.year
    train_df['month'] = train_df['date_time'].dt.month
    train_df['day'] = train_df['date_time'].dt.day
    train_df['hour'] = train_df['date_time'].dt.hour
    train_df['day_of_year'] = train_df['date_time'].dt.dayofyear
    train_df['weekday'] = train_df['date_time'].dt.day_name()
    train_df['date_time'] = train_df['date_time'].dt.date

    lst = [train_df[train_df['building_number'] == i]['power_consumption'].mean() for i in range(1,101)]

    plt.plot(list(range(1,101)), lst)
    plt.xticks([1,10,20,30,40,50,60,70,80,90,100])
    plt.gca().grid(False)
    plt.title('mean consumption per building')
    
    cnt = 0
    for x in sorted(lst, reverse=True)[:3]:
        cnt += 1
        print('top {}: building {}, (mean : {})'.format(str(cnt), str(lst.index(x)), str(int(x))))
    return plt.show()


# In[ ]:





# In[ ]:


# 달 별 24시간 동안 사용량 
def consumption_per_24hours(df):
    plt.plot(df[(df['month'] == 6)].groupby('hour')['power_consumption'].mean(),label='6')
    plt.plot(df[(df['month'] == 7)].groupby('hour')['power_consumption'].mean(),label='7')
    plt.plot(df[(df['month'] == 8)].groupby('hour')['power_consumption'].mean(),label='8')
    plt.legend()
    plt.xticks(list(range(24)))
    plt.title('6~8 consumption per 24 hours')
    return plt.show()


# In[ ]:


# 달 별 에너지 총량
def consumption_per_month(df):
    plt.bar([6,7,8],df.groupby('month')['power_consumption'].mean(),color=['red', 'green', 'blue'])
    plt.xticks([6,7,8])
    plt.title('consumption per each month')
    plt.xlabel('month')
    plt.ylabel('consumption')
    return plt.show()


# In[ ]:


# 1주일 간 에너지 사용 총량
def consumption_per_1week(df):
    data = train_df.groupby('weekday')['power_consumption'].mean()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data = data.sort_index(key=lambda x: x.map({key: idx for idx, key in enumerate(weekday_order)}))
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    return plt.bar(weekday_order,data, color = color_list)


# In[ ]:


# 6/1 ~ 8/24 에너지 사용 총량
def consumption_per_day(df):
    plt.plot(df.groupby('day_of_year')['power_consumption'].mean())
    return plt.show()


# In[ ]:


# 주말, 주중 분리하는 함수
def get_weekend_weekday():
    train_df = pd.read_csv('./train.csv')
    train_df = train_df.fillna(0)
    train_df = train_df.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    })
    train_df.drop('num_date_time', axis = 1, inplace=True)
    
    train_holiday_lst = ['2022-06-04',
                         '2022-06-05',
                         '2022-06-06',
                         '2022-06-11',
                         '2022-06-12',
                         '2022-06-18',
                         '2022-06-19',
                         '2022-06-25',
                         '2022-06-26',
                         '2022-07-02',
                         '2022-07-03',
                         '2022-07-09',
                         '2022-07-10',
                         '2022-07-16',
                         '2022-07-17',
                         '2022-07-23',
                         '2022-07-24',
                         '2022-07-30',
                         '2022-07-31',
                         '2022-08-06',
                         '2022-08-07',
                         '2022-08-13',
                         '2022-08-14',
                         '2022-08-15',
                         '2022-08-20',
                         '2022-08-21']
    from datetime import datetime
    # Convert date strings to datetime.date objects
    train_holiday_lst = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in train_holiday_lst]

    train_df['date_time'] = pd.to_datetime(train_df['date_time'])
    train_df['year'] = train_df['date_time'].dt.year
    train_df['month'] = train_df['date_time'].dt.month
    train_df['day'] = train_df['date_time'].dt.day
    train_df['hour'] = train_df['date_time'].dt.hour
    train_df['day_of_year'] = train_df['date_time'].dt.dayofyear
    train_df['weekday'] = train_df['date_time'].dt.day_name()
    train_df['date_time'] = train_df['date_time'].dt.date
    df = train_df
    weekend_df = df[df['date_time'].isin(train_holiday_lst)]
    weekday_df = df[~(df['date_time'].isin(train_holiday_lst))]
    return weekend_df, weekday_df


# In[ ]:


# 주말, 주중 에너지 사용 총량 비교
def compare_weekday_weekend(df):
    # 주말 데이터와 주중 데이터를 시간대별로 평균 계산 후 그래프로 그리기
    plt.figure(figsize=(12, 6))  # 그림 전체 크기 설정

    # 주말 서브플롯 (1행 2열 중 첫 번째)
    plt.subplot(1, 2, 1)
    plt.plot(weekend_df[(weekend_df['month'] == 6)].groupby('hour')['power_consumption'].mean(), label='weekend6', color='red')
    plt.plot(weekend_df[(weekend_df['month'] == 7)].groupby('hour')['power_consumption'].mean(), label='weekend7', color='blue')
    plt.plot(weekend_df[(weekend_df['month'] == 8)].groupby('hour')['power_consumption'].mean(), label='weekend8', color='green')
    plt.legend()
    plt.xticks(list(range(24)))
    plt.ylim(0, 3500)  # y축 최대값 설정
    plt.title('Weekend Consumption per 24 hours')

    # 주중 서브플롯 (1행 2열 중 두 번째)
    plt.subplot(1, 2, 2)
    plt.plot(weekday_df[(weekday_df['month'] == 6)].groupby('hour')['power_consumption'].mean(), label='weekday6', color='orange')
    plt.plot(weekday_df[(weekday_df['month'] == 7)].groupby('hour')['power_consumption'].mean(), label='weekday7', color='purple')
    plt.plot(weekday_df[(weekday_df['month'] == 8)].groupby('hour')['power_consumption'].mean(), label='weekday8', color='cyan')
    plt.legend()
    plt.xticks(list(range(24)))
    plt.ylim(0, 3700)  # y축 최대값 설정
    plt.title('Weekday Consumption per 24 hours')

    plt.tight_layout()  # 서브플롯 간 간격 조절
    return plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




