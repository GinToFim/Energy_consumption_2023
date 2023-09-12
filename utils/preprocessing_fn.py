"""pandas DataFrame 형태로 주어진 data를 전처리하기 위한 함수들
"""

import numpy as np
import pandas as pd
from typing import *

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """기존 한글로 적혀있던 columns들을 영문으로 변경

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 각 빌딩들의 시간별 정보의 속성이 영문으로 된 DataFrame
        ex. "기온(C)" -> "temp"
    """
    
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


def handling_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 내에 존재하는 결측치를 채운다

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 각 빌딩들의 시간별 정보의 속성 중 결측치를 채운 DataFrame
    """
    
    # 강수량 결측치 0.0으로 채우기
    df["prec"].fillna(0.0, inplace=True)

    # 풍속, 습도 결측치 평균으로 채우고 반올림하기
    df["wind"].fillna(round(df["wind"].mean(), 2), inplace=True)
    df["hum"].fillna(round(df["hum"].mean(), 2), inplace=True)

    return df


def create_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 date열의 값을 이용하여 유용한 열("hour", "day", etc.)을 추가 생성

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: date열의 값을 이용한 유용한 열("hour", "day", etc.)을 추가한 DataFrame 반환
    """
    
    date = pd.to_datetime(df["date"])
    df["date"] = date
    df["hour"] = date.dt.hour
    df["day"] = date.dt.weekday
    df["month"] = date.dt.month
    df["week"] = date.dt.isocalendar().week

    return df


def create_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """주말 이외의 공휴일임을 나타내는 열("holiday") 추가
        (학습시 공휴일임을 인지시키기 위함)

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: "holiday"열이 추가된 DataFrame
    """
    
    ## 공휴일 변수 추가
    df["holiday"] = df.apply(lambda x: 0 if x["day"] < 5 else 1, axis=1)

    # 현충일 6월 6일
    df.loc[(df["date"] >= "2022-06-06") & (df["date"] < "2022-06-07"), "holiday"] = 1

    # # 광복절 8월 15일
    # df.loc[(df['date'] >= '2022-08-15') & (df['date'] < '2022-08-15'), 'holiday'] = 1
    return df


def create_sin_cos_hour(df: pd.DataFrame) -> pd.DataFrame:
    """24시간제로 표현한 "hour"을 주기가 1/24인 주기함수(sin, cos)에 대입한 값을 저정하는 "sin_hour"과 "cos_hour"열을 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: "sin_hour"과 "cos_hour"열을 추가한 DataFrame
    """
    
    # sin & cos 변수 추가
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_temp_f(df: pd.DataFrame) -> pd.DataFrame:
    """화씨 온도 표시법에 해당하는 온도 값을 나타내는 열("temp_f")을 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 화씨 온도 표시법에 해당하는 온도 값을 나타내는 열("temp_f")을 추가한 DataFrame
    """
    
    # 화씨 온도 추가
    df["temp_f"] = (df["temp"] * 9 / 5) + 32
    return df


def create_wind_chill_temp(df: pd.DataFrame) -> pd.DataFrame:
    """ "temp", "wind"을 이용하여 체감온도를 나타내는 열("wind_chill_temp") 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: "temp", "wind"을 이용하여 체감온도를 나타내는 열("wind_chill_temp") 추가한 DataFrame
    """
    
    # 체감 온도 변수 추가
    # https://www.weather.go.kr/w/theme/daily-life/regional-composite-index.do
    df["wind_chill_temp"] = (
        13.12
        + 0.6215 * df["temp"]
        - 11.37 * (df["wind"] * 3.6) ** 0.16
        + 0.3965 * (df["wind"] * 3.6) ** 0.16 * df["temp"]
    )
    return df


def create_thi(df: pd.DataFrame) -> pd.DataFrame:
    """ "temp", "hum"을 이용하여 불쾌지수와 관련이 있는 온도 습도 지수 열("THI") 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 온도 습도 지수 열("THI") 추가한 DataFrame
    """
    
    # Temperature Humidity Index(THI) 변수 추가
    df["THI"] = (
        9 / 5 * df["temp"]
        - 0.55 * (1 - df["hum"] / 100) * (9 / 5 * df["temp"] - 26)
        + 32
    )
    return df


def create_cdh(df: pd.DataFrame) -> pd.DataFrame:
    """CDH 함수를 이용하여 Cooling Degree Hour 값이 들어있는 1차원 numpy 배열을 생성하고
        이를 새로운 열("CDH") 생성

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 새로운 열("CDH")이 추가된 DataFrame
    """
    
    # Cooling Degree Hour 변수 추가
    def CDH(xs: np.ndarray) -> np.ndarray:
        """1차원 numpy 배열을 입력받아 Cooling Degree Hour 변수가 들어있는 numpy 배열을 반환

        Args:
            xs (np.ndarray): 특정 빌딩의 온도 값을 저장하는 1차원 numpy 배열

        Returns:
            np.ndarray: 해당 빌딩의 Cooling Degree Hour 값을 저장하는 1차원 numpy 배열
        """
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


def create_working_hour(df: pd.DataFrame) -> pd.DataFrame:
    """각 빌딩들의 시간별로의 정보를 담고 있는 DataFrame을 입력받고,
        시간대를 3가지로 분류할 수 있도록
        "work_hour", "lunch_hour", "lunch_hour2" 3가지 열을 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: "work_hour", "lunch_hour", "lunch_hour2" 열이 추가된 DataFrame
    """
    
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


def create_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    """기존 DataFrame에 열지수("heat_index") 열을 추가

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 열지수("heat_index") 열이 추가된 DataFrame
    """
    
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
def create_day_hour_mean(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 입력받아 시간별, 요일별 발전량 평균값을 저장하는 DataFrame을 만들어 반환

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 시간별, 요일별 발전량 평균값을 저장하는 DataFrame
    """
    
    day_hour_power_mean = pd.pivot_table(df, values = 'power', 
                                         index = ['building_num', 'hour', 'day'], 
                                         aggfunc = np.mean).reset_index()
    
    day_hour_power_mean.columns = ['building_num', 'hour', 'day', 'day_hour_mean']
    
    return day_hour_power_mean

def create_day_hour_std(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 입력받아 시간별, 요일별 발전량의 표준편차를 저장하는 DataFrame을 만들어 반환

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 시간별, 요일별 발전량의 표준편차를 저장하는 DataFrame
    """
    
    day_hour_power_std = pd.pivot_table(df, values = 'power', 
                                         index = ['building_num', 'hour', 'day'], 
                                         aggfunc = np.std).reset_index()
    
    day_hour_power_std.columns = ['building_num', 'hour', 'day', 'day_hour_std']
    
    return day_hour_power_std


def create_hour_mean(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 입력받아 시간별 발전량 평균값을 저장하는 DataFrame을 만들어 반환

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 시간별 발전량 평균값을 저장하는 DataFrame
    """
    
    hour_power_mean = pd.pivot_table(df, values = 'power', 
                                         index = ['building_num', 'hour'], 
                                         aggfunc = np.mean).reset_index()
    
    hour_power_mean.columns = ['building_num', 'hour', 'hour_mean']
    
    return hour_power_mean

def create_hour_std(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame을 입력받아 시간별 발전량 표준편차를 저장하는 DataFrame을 만들어 반환

    Args:
        df (pd.DataFrame): 각 빌딩들의 시간별 정보를 담고 있는 DataFrame

    Returns:
        pd.DataFrame: 시간별 발전량 표준편차를 저장하는 DataFrame
    """
    
    hour_power_std = pd.pivot_table(df, values = 'power', 
                                         index = ['building_num', 'hour'], 
                                         aggfunc = np.std).reset_index()
    
    hour_power_std.columns = ['building_num', 'hour', 'hour_std']
    
    return hour_power_std