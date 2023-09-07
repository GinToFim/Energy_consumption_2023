import numpy as np
import pandas as pd


def rename_columns(df):
    # 컬럼명 영어로 수정
    df_cols = [
        "num_date_time",
        "building_num",
        "date",
        "temp",
        "prec",
        "wind",
        "hum",
        "sunshine",
        "solar",
        "power",
    ]

    df_cols_dict = {key: value for key, value in zip(df.columns, df_cols)}

    df = df.rename(columns=df_cols_dict)

    return df


def handling_missing_values(df):
    # 강수량 결측치 0.0으로 채우기
    df["prec"].fillna(0.0, inplace=True)

    # 풍속, 습도 결측치 평균으로 채우고 반올림하기
    df["wind"].fillna(round(df["wind"].mean(), 2), inplace=True)
    df["hum"].fillna(round(df["hum"].mean(), 2), inplace=True)

    return df


def create_time_columns(df):
    date = pd.to_datetime(df["date"])
    df["date"] = date
    df["hour"] = date.dt.hour
    df["day"] = date.dt.weekday
    df["month"] = date.dt.month
    df["week"] = date.dt.isocalendar().week

    return df


def create_holiday(df):
    ## 공휴일 변수 추가
    df["holiday"] = df.apply(lambda x: 0 if x["day"] < 5 else 1, axis=1)

    # 현충일 6월 6일
    df.loc[(df["date"] >= "2022-06-06") & (df["date"] < "2022-06-07"), "holiday"] = 1

    # # 광복절 8월 15일
    # df.loc[(df['date'] >= '2022-08-15') & (df['date'] < '2022-08-15'), 'holiday'] = 1
    return df


def create_sin_cos_hour(df):
    # sin & cos 변수 추가
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_temp_f(df):
    # 화씨 온도 추가
    df["temp_f"] = (df["temp"] * 9 / 5) + 32
    return df


def create_wind_chill_temp(df):
    # 체감 온도 변수 추가
    # https://www.weather.go.kr/w/theme/daily-life/regional-composite-index.do
    df["wind_chill_temp"] = (
        13.12
        + 0.6215 * df["temp"]
        - 11.37 * (df["wind"] * 3.6) ** 0.16
        + 0.3965 * (df["wind"] * 3.6) ** 0.16 * df["temp"]
    )
    return df


def create_thi(df):
    # Temperature Humidity Index(THI) 변수 추가
    df["THI"] = (
        9 / 5 * df["temp"]
        - 0.55 * (1 - df["hum"] / 100) * (9 / 5 * df["temp"] - 26)
        + 32
    )
    return df


def create_cdh(df):
    # Cooling Degree Hour 변수 추가
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[: (i + 1)] - 26))
            else:
                ys.append(np.sum(xs[(i - 11) : (i + 1)] - 26))
        return np.array(ys)

    cdhs = np.array([])
    for num in range(1, 101):
        building_df = df[df["building_num"] == num]
        cdh = CDH(building_df["temp"].values)
        cdhs = np.concatenate([cdhs, cdh])

    df["CDH"] = cdhs

    return df


def create_working_hour(df):
    # 일 관련 시간 추가
    df["work_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 19)).astype(int)
    df["lunch_hour"] = (
        (df["hour"] >= 11) & (df["hour"] <= 13) & (df["day"] <= 4)
    ).astype(int)
    df["lunch_hour2"] = (
        (df["hour"] >= 12) & (df["hour"] <= 14) & (df["day"] > 4)
    ).astype(int)

    df["dinner_hour"] = ((df["hour"] >= 17) & (df["hour"] <= 22)).astype(int)
    df["dinner_hour2"] = (
        (df["hour"] >= 18) & (df["day"] >= 4) & (df["day"] <= 5)
    ).astype(int)

    return df


def create_heat_index(df):
    T = df["temp_f"]
    RH = df["hum"]
    HI = pd.Series([0] * len(T), name="heat_index")

    condition3 = T < 80
    condition4 = T >= 80
    condition1 = (RH > 85) & ((T > 80) & (T < 87))
    condition2 = (RH < 13) & ((T > 80) & (T < 112))

    HI[condition3] = 0.5 * (
        T[condition3] + 61.0 + ((T[condition3] - 68.0) * 1.2) + (RH[condition3] * 0.094)
    )
    HI[condition4] = (
        -42.379
        + 2.04901523 * T[condition4]
        + 10.14333127 * RH[condition4]
        - 0.22475541 * T[condition4] * RH[condition4]
        - 0.00683783 * T[condition4] * T[condition4]
        - 0.05481717 * RH[condition4] * RH[condition4]
        + 0.00122874 * T[condition4] * T[condition4] * RH[condition4]
        + 0.00085282 * T[condition4] * RH[condition4] * RH[condition4]
        - 0.00000199 * T[condition4] * T[condition4] * RH[condition4] * RH[condition4]
    )
    HI[condition1] = HI[condition1] + ((RH[condition1] - 85) / 10) * (
        (87 - T[condition1]) / 5
    )
    HI[condition2] = HI[condition2] - ((13 - RH[condition2]) / 4) * np.sqrt(
        (17 - abs(T[condition2] - 95.0)) / 17
    )

    df["Heat_index"] = HI

    return df

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