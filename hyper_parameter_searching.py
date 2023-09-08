import random
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# data pre-processing load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# others ML model load
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm

# For reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # seed 고정


# In[38]:


# Load 데이터프레임

train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train_split1 .csv')
valid_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/valid_split1 .csv')


# In[39]:


train_df


# In[40]:


def rename_columns(df):
    # 컬럼명 영어로 수정
    df_cols = ['num_date_time', 'building_num', 'date', 'temp',
               'prec', 'wind', 'hum', 'sunshine', 'solar', 'power']

    df_cols_dict = {key: value for key, value in zip(df.columns, df_cols)}

    df = df.rename(columns=df_cols_dict)

    return df

def handling_missing_values(df):
    # 강수량 결측치 0.0으로 채우기
    df['prec'].fillna(0.0, inplace=True)

    # 풍속, 습도 결측치 평균으로 채우고 반올림하기
    df['wind'].fillna(round(df['wind'].mean(),2), inplace=True)
    df['hum'].fillna(round(df['hum'].mean(),2), inplace=True)

    return df

def create_time_columns(df):
    date = pd.to_datetime(df['date'])
    df['date'] = date
    df['hour'] = date.dt.hour
    df['day'] =  date.dt.weekday
    df['month'] = date.dt.month
    df['week'] = date.dt.isocalendar().week

    return df

def create_holiday(df):
    ## 공휴일 변수 추가
    df['holiday'] = df.apply(lambda x : 0 if x['day'] < 5 else 1, axis = 1)

    # 현충일 6월 6일
    df.loc[(df['date'] >= '2022-06-06') & (df['date'] < '2022-06-07'), 'holiday'] = 1

    # 광복절 8월 15일
    df.loc[(df['date'] >= '2022-08-15') & (df['date'] < '2022-08-15'), 'holiday'] = 1

    return df

def create_sin_cos_hour(df):
    # sin & cos 변수 추가
    df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)

    return df

def create_temp_f(df):
    # 화씨 온도 추가
    df['temp_f'] = (df['temp'] * 9/5) + 32

    return df

def create_wind_chill_temp(df):
    # 체감 온도 변수 추가
    # https://www.weather.go.kr/w/theme/daily-life/regional-composite-index.do
    df['wind_chill_temp'] = 13.12 + 0.6215*df['temp'] - 11.37*(df['wind']*3.6)**0.16 + 0.3965*(df['wind']*3.6)**0.16*df['temp']

    return df

# 열지수 column 추가
# https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
def create_Heat_index(df):
    T = df['temp_f']
    RH = df['hum']
    HI = pd.Series([0] * len(T), name = 'Heat_index')

    condition3 = T < 80
    condition4 = T >= 80
    condition1 = (RH > 85) & ((T > 80) & (T < 87))
    condition2 = (RH < 13) & ((T > 80) & (T < 112))

    HI[condition3] = 0.5 * (T[condition3] + 61.0 + ((T[condition3]-68.0)*1.2) + (RH[condition3]*0.094))

    HI[condition4] = -42.379 + 2.04901523*T[condition4] + 10.14333127*RH[condition4] - .22475541*T[condition4]*RH[condition4] - .00683783*T[condition4]*T[condition4] - .05481717*RH[condition4]*RH[condition4] + .00122874*T[condition4]*T[condition4]*RH[condition4] + .00085282*T[condition4]*RH[condition4]*RH[condition4] - .00000199*T[condition4]*T[condition4]*RH[condition4]*RH[condition4]

    HI[condition1] = HI[condition1] + ((RH[condition1]-85)/10) * ((87-T[condition1])/5)

    HI[condition2] = HI[condition2] - ((13-RH[condition2])/4)*np.sqrt((17-abs(T[condition2]-95.))/17)

    df['Heat_index'] = HI

    return df


def create_thi(df):
    # Temperature Humidity Index(THI) 변수 추가
    df['THI'] = 9/5*df['temp'] - 0.55*(1-df['hum']/100)*(9/5*df['temp']-26)+32

    return df

def create_cdh(df):
    # Cooling Degree Hour 변수 추가
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i+1)]-26))
            else:
                ys.append(np.sum(xs[(i-11):(i+1)]-26))
        return np.array(ys)

    cdhs = np.array([])
    for num in range(1,101):
        building_df = df[df['building_num'] == num]
        cdh = CDH(building_df['temp'].values)
        cdhs = np.concatenate([cdhs, cdh])

    df['CDH'] = cdhs

    return df

def create_working_hour(df):
    # 일 관련 시간 추가
    df['work_hour'] = ((df['hour']>=8) & (df['hour']<=19)).astype(int)
    df['lunch_hour'] = ((df['hour']>=11) & (df['hour']<=13) & (df['day']<=4)).astype(int)
    df['lunch_hour2'] = ((df['hour']>=12) & (df['hour']<=14) & (df['day']>4)).astype(int)

    df['dinner_hour'] = ((df['hour']>=17) & (df['hour']<=22)).astype(int)
    df['dinner_hour2'] = ((df['hour']>=18) & (df['day']>=4) & (df['day']<=5)).astype(int)

    return df


# In[41]:


### 발전량 평균 넣어주기

## 건물당 요일 + 시간별 발전량 평균 : day_hour_mean
def create_day_hour_mean(df):
    day_hour_power_mean = pd.pivot_table(df, values = 'power',
                                         index = ['building_num', 'hour', 'day'],
                                         aggfunc = np.mean).reset_index()

    day_hour_power_mean.columns = ['building_num', 'hour', 'day', 'day_hour_mean']

    return day_hour_power_mean

def create_day_hour_std(df):
    day_hour_power_std = pd.pivot_table(df, values = 'power',
                                         index = ['building_num', 'hour', 'day'],
                                         aggfunc = np.std).reset_index()

    day_hour_power_std.columns = ['building_num', 'hour', 'day', 'day_hour_std']

    return day_hour_power_std


def create_hour_mean(df):
    hour_power_mean = pd.pivot_table(df, values = 'power',
                                         index = ['building_num', 'hour'],
                                         aggfunc = np.mean).reset_index()

    hour_power_mean.columns = ['building_num', 'hour', 'hour_mean']

    return hour_power_mean

def create_hour_std(df):
    hour_power_std = pd.pivot_table(df, values = 'power',
                                         index = ['building_num', 'hour'],
                                         aggfunc = np.std).reset_index()

    hour_power_std.columns = ['building_num', 'hour', 'hour_std']

    return hour_power_std


# In[42]:


from collections import defaultdict

def create_cluster(df):
    weather_cluster = defaultdict(list)

    first_date = df.iloc[0]['date']
    tmp = df[df['date']==first_date]

    for idx, (temp, wind, hum) in enumerate(zip(tmp['temp'], tmp['wind'], tmp['hum'])):
        weather_cluster[(temp, wind, hum)].append(idx + 1)

    df['cluster'] = 0

    for idx, key in enumerate(weather_cluster) :
        df.loc[df['building_num'].isin(weather_cluster[key]), 'cluster'] = idx + 1

    return df

def create_cluster_hour_mean(df):
    cluster_hour_power_mean = pd.pivot_table(df, values = 'power',
                                         index = ['cluster', 'hour', 'day'],
                                         aggfunc = np.mean).reset_index()

    cluster_hour_power_mean.columns = ['cluster', 'hour', 'day', 'cluster_hour_mean']
    return cluster_hour_power_mean

def create_cluster_hour_std(df):
    cluster_hour_power_std = pd.pivot_table(df, values = 'power',
                                         index = ['cluster', 'hour', 'day'],
                                         aggfunc = np.std).reset_index()

    cluster_hour_power_std.columns = ['cluster', 'hour', 'day', 'cluster_hour_std']
    return cluster_hour_power_std


# In[43]:


def train_pre_processing(train_df):
    # 컬럼명 영어로 수정
    train_df = rename_columns(train_df)

    # 결측치 채우기
    train_df = handling_missing_values(train_df)

    # 시계열 - 시간 관련 변수들 생성
    train_df = create_time_columns(train_df)
    train_df = create_working_hour(train_df)

    ## 공휴일 변수 추가
    train_df = create_holiday(train_df)

    # sin & cos 변수 추가
    train_df = create_sin_cos_hour(train_df)

    # 화씨 온도 변수 추가
    train_df = create_temp_f(train_df)

    # 체감 온도 변수 추가
    train_df = create_wind_chill_temp(train_df)

    # 열지수(Heat index) 변수 추가
    train_df = create_Heat_index(train_df)

    # Temperature Humidity Index(THI) 변수 추가
    train_df = create_thi(train_df)

    # Cooling Degree Hour(CDH) 변수 추가
    train_df = create_cdh(train_df)

    #  day_hour_mean 변수 추가
    day_hour_power_mean = create_day_hour_mean(train_df)
    train_df = pd.merge(train_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])

    #  day_hour_std 변수 추가
    day_hour_power_std = create_day_hour_std(train_df)
    train_df = pd.merge(train_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])

    # cluster 변수 추가
    train_df = create_cluster(train_df)

    cluster_hour_power_mean = create_cluster_hour_mean(train_df)
    train_df = pd.merge(train_df, cluster_hour_power_mean, how='left', on=['cluster', 'hour', 'day'])

    cluster_hour_power_std = create_cluster_hour_std(train_df)
    train_df = pd.merge(train_df, cluster_hour_power_std, how='left', on=['cluster', 'hour', 'day'])


    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    train_df = train_df.drop(['num_date_time', 'date', 'sunshine', 'solar', 'hour'], axis=1)

    return train_df


# In[44]:


def test_pre_processing(test_df, train_df=None):
    # 컬럼명 영어로 수정
    test_df = rename_columns(test_df)
    train_df = rename_columns(train_df)

    # 결측치 채우기
    test_df = handling_missing_values(test_df)

    # 시계열 - 시간 관련 변수들 생성
    test_df = create_time_columns(test_df)
    train_df = create_time_columns(train_df)

    test_df = create_working_hour(test_df)

    ## 공휴일 변수 추가
    test_df = create_holiday(test_df)

    # sin & cos 변수 추가
    test_df = create_sin_cos_hour(test_df)

    # 화씨 온도 변수 추가
    test_df = create_temp_f(test_df)

    # 체감 온도 변수 추가
    test_df = create_wind_chill_temp(test_df)

    # 열지수(Heat index) 변수 추가
    test_df = create_Heat_index(test_df)

    # Temperature Humidity Index(THI) 변수 추가
    test_df = create_thi(test_df)

    # Cooling Degree Hour(CDH) 변수 추가
    test_df = create_cdh(test_df)

    #  day_hour_mean 변수 추가
    day_hour_power_mean = create_day_hour_mean(train_df)
    test_df = pd.merge(test_df, day_hour_power_mean, how='left', on=['building_num', 'hour', 'day'])

    #  day_hour_std 변수 추가
    day_hour_power_std = create_day_hour_std(train_df)
    test_df = pd.merge(test_df, day_hour_power_std, how='left', on=['building_num', 'hour', 'day'])


    # cluster 변수 추가
    train_df = create_cluster(train_df)
    test_df = create_cluster(test_df)

    cluster_hour_power_mean = create_cluster_hour_mean(train_df)
    test_df = pd.merge(test_df, cluster_hour_power_mean, how='left', on=['cluster', 'hour', 'day'])

    cluster_hour_power_std = create_cluster_hour_std(train_df)
    test_df = pd.merge(test_df, cluster_hour_power_std, how='left', on=['cluster', 'hour', 'day'])


    # 컬럼 제거
    # num_date_time, date, sunshine, solar, hour drop 컬럼 제거
    if 'sunshine' in test_df.columns.tolist() :
        test_df = test_df.drop(['sunshine', 'solar'], axis=1)

    test_df = test_df.drop(['num_date_time', 'date', 'hour'], axis=1)

    return test_df


# In[45]:


dp_train_df = train_pre_processing(train_df)
dp_train_df


# In[46]:


dp_valid_df = test_pre_processing(valid_df, train_df)
dp_valid_df


# In[47]:


# 주어진 1차원 리스트들의 리스트
one_dimensional_lists = dp_train_df['prec']

# 1차원 리스트를 24개씩 잘라서 2차원 리스트로 구성
num_elements_per_row = 24
num_rows = len(one_dimensional_lists) // num_elements_per_row
two_dimensional_lists = [one_dimensional_lists[i * num_elements_per_row:(i + 1) * num_elements_per_row] for i in range(num_rows)]

lst = []
for x in two_dimensional_lists:
    lst.append(x.interpolate(method='polynomial', order=3))

wis = []
for x in lst:
    wis.extend(x)

dp_train_df['prec'] = wis
dp_train_df


# In[48]:


# 주어진 1차원 리스트들의 리스트
one_dimensional_lists = dp_valid_df['prec']

# 1차원 리스트를 24개씩 잘라서 2차원 리스트로 구성
num_elements_per_row = 24
num_rows = len(one_dimensional_lists) // num_elements_per_row
two_dimensional_lists = [one_dimensional_lists[i * num_elements_per_row:(i + 1) * num_elements_per_row] for i in range(num_rows)]

lst = []
for x in two_dimensional_lists:
    lst.append(x.interpolate(method='polynomial', order=3))

wis = []
for x in lst:
    wis.extend(x)

dp_valid_df['prec'] = wis
dp_valid_df


# In[49]:


dic ={}
for i in range(1, 101):
    dic['df'+str(i)] = dp_valid_df[dp_valid_df['building_num'] == i][['temp', 'prec', 'wind', 'hum']].reset_index(drop = True)


# In[50]:


# DataFrame 그룹핑 함수
def group_equal_dataframes(dataframes):
    grouped = []  # 그룹화된 DataFrame 리스트
    grouped_indices = set()

    for i, df1 in enumerate(dataframes):
        if i in grouped_indices:
            continue

        group = [i]

        for j, df2 in enumerate(dataframes[i + 1:], start=i + 1):
            if j not in grouped_indices and dic[df1].equals(dic[df2]):
                group.append(j)
                grouped_indices.add(j)

        if len(group) > 1:
            grouped.append(group)

    return grouped

# DataFrame 그룹핑
dataframes_to_group = list(dic.keys())
grouped_dataframes = group_equal_dataframes(dataframes_to_group)

def add_one_to_2d_list(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] += 1
    return matrix

grouped_dataframes = add_one_to_2d_list(grouped_dataframes)

# 결과 출력
for group_id, group in enumerate(grouped_dataframes, start=1):
    print(f"Group {group_id}: {group}")


# In[51]:


dic = {}
for idx, group in enumerate(grouped_dataframes):
    dic[int(idx+1)] = group


# In[52]:


original_dict = dic

flipped_dict = {value: key for key, values in original_dict.items() for value in values}
print(flipped_dict)


# In[53]:


total = set(list(range(1,101)))
grouped = set(list(flipped_dict.keys()))
non_grouped = total - grouped


# In[54]:


dic1={}
for x in non_grouped:
    dic1[int(x)] = 0


# In[55]:


flipped_dict.update(dic1)
flipped_dict


# In[56]:


final = {key: values for key, values in sorted(flipped_dict.items())}


# In[57]:


dp_train_df['cluster'] = dp_train_df['building_num'].map(final)
dp_train_df.drop(['cluster_hour_mean','cluster_hour_std'], axis = 1, inplace = True)


# In[58]:


dp_valid_df['cluster'] = dp_valid_df['building_num'].map(final)
dp_valid_df.drop(['cluster_hour_mean','cluster_hour_std'], axis = 1, inplace = True)
dp_valid_df


# In[59]:


# Define SMAPE loss function
def SMAPE(actual, pred):
    return 100 * np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred)))


# In[60]:


#### alpha를 argument로 받는 함수로 실제 objective function을 wrapping하여 alpha값을 쉽게 조정할 수 있도록 작성했습니다.
# custom objective function for forcing model not to underestimate
def weighted_mse(alpha = 1):
    def weighted_mse_fixed(label, pred):
        residual = (label - pred).astype("float")
        grad = np.where(residual>0, -2*alpha*residual, -2*residual)
        hess = np.where(residual>0, 2*alpha, 2.0)
        return grad, hess
    return weighted_mse_fixed


# In[61]:


# n_estimator 때문에 너무 오래 걸려서 100으로 고정 시키고 나머지 파라미터 서칭 후 따로 n_estimator 찾을 예정


# In[66]:


from sklearn.model_selection import PredefinedSplit, GridSearchCV
df = pd.DataFrame(columns = ['n_estimators', 'eta', 'min_child_weight','max_depth', 'colsample_bytree', 'subsample'])
preds = np.array([])

grid = {'n_estimators' : [100], 'eta' : [0.01], 'min_child_weight' : np.arange(1, 8, 1),
        'max_depth' : np.arange(3,9,1) , 'colsample_bytree' :np.arange(0.8, 1.0, 0.1),
        'subsample' :np.arange(0.8, 1.0, 0.1)} # fix the n_estimators & eta(learning rate)

outlying_building = [3,14,21,30,40,42,53,54,91,95,98]

for i in tqdm(outlying_building):
    X_train = dp_train_df[dp_train_df["building_num"] == i].drop(['building_num', 'power'], axis=1)
    y_train = dp_train_df[dp_train_df["building_num"] == i]['power']

    X_valid = dp_valid_df[dp_valid_df["building_num"] == i].drop(['building_num', 'power'], axis=1)
    y_valid = dp_valid_df[dp_valid_df["building_num"] == i]['power']

    X_train['week'] = X_train['week'].astype('int64')
    X_valid['week'] = X_valid['week'].astype('int64')


    #pds = PredefinedSplit(np.append(-np.ones(len(x)-168), np.zeros(168)))
    gcv = GridSearchCV(estimator = XGBRegressor(tree_method = 'hist',
                                                objective = weighted_mse(1)),
                       param_grid = grid, scoring = SMAPE, cv = 5, refit = True, verbose = True)


    gcv.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    best = gcv.best_estimator_
    params = gcv.best_params_
    print(params)
    pred = best.predict(X_valid)
    building = 'building '+str(i)
    print(building + ' || SMAPE : {}'.format(SMAPE(y_valid, pred)))
    preds = np.append(preds, pred)
    df_1 = pd.concat([df, pd.DataFrame(params, index = [0])], axis = 0)
df_1.to_csv('/content/drive/MyDrive/Colab Notebooks/hyperparameter_xgb(1~50).csv', index = False) # save the tuned parameters


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




