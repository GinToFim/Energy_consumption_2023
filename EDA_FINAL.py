{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4ebeec6c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import os\n",
    "import math\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# data pre-processing load\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# others ML model load\n",
    "from lightgbm import LGBMRegressor\n",
    "from xgboost import XGBRegressor\n",
    "from catboost import CatBoostRegressor\n",
    "\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0ebe5115",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For reproducibility\n",
    "\n",
    "def seed_everything(seed):\n",
    "    random.seed(seed)\n",
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
    "    np.random.seed(seed)\n",
    "    \n",
    "seed_everything(42) # seed 고정"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5271988",
   "metadata": {},
   "outputs": [],
   "source": [
    "# path 설정\n",
    "\n",
    "dir_path = \"./data/energy_cosumption\"\n",
    "\n",
    "train_path = os.path.join(dir_path, \"train_split.csv\")\n",
    "original_train_path = os.path.join(dir_path, \"train.csv\")\n",
    "\n",
    "valid_path = os.path.join(dir_path, \"valid_split.csv\")\n",
    "\n",
    "test_path = os.path.join(dir_path, \"test.csv\")\n",
    "\n",
    "building_path = os.path.join(dir_path, \"building_info.csv\")\n",
    "submission_path = os.path.join(dir_path, \"sample_submission.csv\") "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "367c7295",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load 데이터프레임\n",
    "\n",
    "train_df = pd.read_csv(train_path)\n",
    "original_train_df = pd.read_csv(original_train_path)\n",
    "\n",
    "valid_df = pd.read_csv(valid_path)\n",
    "\n",
    "test_df = pd.read_csv(test_path)\n",
    "\n",
    "building_df = pd.read_csv(building_path)\n",
    "submission_df = pd.read_csv(submission_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c5e31cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 빌딩 별 전력사용량 평균\n",
    "def mean_of_power_per_building(df):\n",
    "    train_df = pd.read_csv('./train.csv')\n",
    "    train_df = train_df.rename(columns={\n",
    "        '건물번호': 'building_number',\n",
    "        '일시': 'date_time',\n",
    "        '기온(C)': 'temperature',\n",
    "        '강수량(mm)': 'rainfall',\n",
    "        '풍속(m/s)': 'windspeed',\n",
    "        '습도(%)': 'humidity',\n",
    "        '일조(hr)': 'sunshine',\n",
    "        '일사(MJ/m2)': 'solar_radiation',\n",
    "        '전력소비량(kWh)': 'power_consumption'\n",
    "    })\n",
    "    train_df = train_df.fillna(0)\n",
    "    train_df.drop('num_date_time', axis = 1, inplace=True)\n",
    "\n",
    "    train_df['date_time'] = pd.to_datetime(train_df['date_time'])\n",
    "    train_df['year'] = train_df['date_time'].dt.year\n",
    "    train_df['month'] = train_df['date_time'].dt.month\n",
    "    train_df['day'] = train_df['date_time'].dt.day\n",
    "    train_df['hour'] = train_df['date_time'].dt.hour\n",
    "    train_df['day_of_year'] = train_df['date_time'].dt.dayofyear\n",
    "    train_df['weekday'] = train_df['date_time'].dt.day_name()\n",
    "    train_df['date_time'] = train_df['date_time'].dt.date\n",
    "\n",
    "    lst = [train_df[train_df['building_number'] == i]['power_consumption'].mean() for i in range(1,101)]\n",
    "\n",
    "    plt.plot(list(range(1,101)), lst)\n",
    "    plt.xticks([1,10,20,30,40,50,60,70,80,90,100])\n",
    "    plt.gca().grid(False)\n",
    "    plt.title('mean consumption per building')\n",
    "    \n",
    "    cnt = 0\n",
    "    for x in sorted(lst, reverse=True)[:3]:\n",
    "        cnt += 1\n",
    "        print('top {}: building {}, (mean : {})'.format(str(cnt), str(lst.index(x)), str(int(x))))\n",
    "    return plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "974fd396",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a5305ac3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 달 별 24시간 동안 사용량 \n",
    "def consumption_per_24hours(df):\n",
    "    plt.plot(df[(df['month'] == 6)].groupby('hour')['power_consumption'].mean(),label='6')\n",
    "    plt.plot(df[(df['month'] == 7)].groupby('hour')['power_consumption'].mean(),label='7')\n",
    "    plt.plot(df[(df['month'] == 8)].groupby('hour')['power_consumption'].mean(),label='8')\n",
    "    plt.legend()\n",
    "    plt.xticks(list(range(24)))\n",
    "    plt.title('6~8 consumption per 24 hours')\n",
    "    return plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b7c74491",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fe4349bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 달 별 에너지 총량\n",
    "def consumption_per_month(df):\n",
    "    plt.bar([6,7,8],df.groupby('month')['power_consumption'].mean(),color=['red', 'green', 'blue'])\n",
    "    plt.xticks([6,7,8])\n",
    "    plt.title('consumption per each month')\n",
    "    plt.xlabel('month')\n",
    "    plt.ylabel('consumption')\n",
    "    return plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c46ea514",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "679d55f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1주일 간 에너지 사용 총량\n",
    "def consumption_per_1week(df):\n",
    "    data = train_df.groupby('weekday')['power_consumption'].mean()\n",
    "    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
    "    data = data.sort_index(key=lambda x: x.map({key: idx for idx, key in enumerate(weekday_order)}))\n",
    "    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']\n",
    "    return plt.bar(weekday_order,data, color = color_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9782144",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6/1 ~ 8/24 에너지 사용 총량\n",
    "def consumption_per_day(df):\n",
    "    plt.plot(df.groupby('day_of_year')['power_consumption'].mean())\n",
    "    return plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad025978",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a3e33b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 주말, 주중 분리하는 함수\n",
    "def get_weekend_weekday():\n",
    "    train_df = pd.read_csv('./train.csv')\n",
    "    train_df = train_df.fillna(0)\n",
    "    train_df = train_df.rename(columns={\n",
    "        '건물번호': 'building_number',\n",
    "        '일시': 'date_time',\n",
    "        '기온(C)': 'temperature',\n",
    "        '강수량(mm)': 'rainfall',\n",
    "        '풍속(m/s)': 'windspeed',\n",
    "        '습도(%)': 'humidity',\n",
    "        '일조(hr)': 'sunshine',\n",
    "        '일사(MJ/m2)': 'solar_radiation',\n",
    "        '전력소비량(kWh)': 'power_consumption'\n",
    "    })\n",
    "    train_df.drop('num_date_time', axis = 1, inplace=True)\n",
    "    \n",
    "    train_holiday_lst = ['2022-06-04',\n",
    "                         '2022-06-05',\n",
    "                         '2022-06-06',\n",
    "                         '2022-06-11',\n",
    "                         '2022-06-12',\n",
    "                         '2022-06-18',\n",
    "                         '2022-06-19',\n",
    "                         '2022-06-25',\n",
    "                         '2022-06-26',\n",
    "                         '2022-07-02',\n",
    "                         '2022-07-03',\n",
    "                         '2022-07-09',\n",
    "                         '2022-07-10',\n",
    "                         '2022-07-16',\n",
    "                         '2022-07-17',\n",
    "                         '2022-07-23',\n",
    "                         '2022-07-24',\n",
    "                         '2022-07-30',\n",
    "                         '2022-07-31',\n",
    "                         '2022-08-06',\n",
    "                         '2022-08-07',\n",
    "                         '2022-08-13',\n",
    "                         '2022-08-14',\n",
    "                         '2022-08-15',\n",
    "                         '2022-08-20',\n",
    "                         '2022-08-21']\n",
    "    from datetime import datetime\n",
    "    # Convert date strings to datetime.date objects\n",
    "    train_holiday_lst = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in train_holiday_lst]\n",
    "\n",
    "    train_df['date_time'] = pd.to_datetime(train_df['date_time'])\n",
    "    train_df['year'] = train_df['date_time'].dt.year\n",
    "    train_df['month'] = train_df['date_time'].dt.month\n",
    "    train_df['day'] = train_df['date_time'].dt.day\n",
    "    train_df['hour'] = train_df['date_time'].dt.hour\n",
    "    train_df['day_of_year'] = train_df['date_time'].dt.dayofyear\n",
    "    train_df['weekday'] = train_df['date_time'].dt.day_name()\n",
    "    train_df['date_time'] = train_df['date_time'].dt.date\n",
    "    df = train_df\n",
    "    weekend_df = df[df['date_time'].isin(train_holiday_lst)]\n",
    "    weekday_df = df[~(df['date_time'].isin(train_holiday_lst))]\n",
    "    return weekend_df, weekday_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c3d3ebc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "223c59ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 주말, 주중 에너지 사용 총량 비교\n",
    "def compare_weekday_weekend(df):\n",
    "    # 주말 데이터와 주중 데이터를 시간대별로 평균 계산 후 그래프로 그리기\n",
    "    plt.figure(figsize=(12, 6))  # 그림 전체 크기 설정\n",
    "\n",
    "    # 주말 서브플롯 (1행 2열 중 첫 번째)\n",
    "    plt.subplot(1, 2, 1)\n",
    "    plt.plot(weekend_df[(weekend_df['month'] == 6)].groupby('hour')['power_consumption'].mean(), label='weekend6', color='red')\n",
    "    plt.plot(weekend_df[(weekend_df['month'] == 7)].groupby('hour')['power_consumption'].mean(), label='weekend7', color='blue')\n",
    "    plt.plot(weekend_df[(weekend_df['month'] == 8)].groupby('hour')['power_consumption'].mean(), label='weekend8', color='green')\n",
    "    plt.legend()\n",
    "    plt.xticks(list(range(24)))\n",
    "    plt.ylim(0, 3500)  # y축 최대값 설정\n",
    "    plt.title('Weekend Consumption per 24 hours')\n",
    "\n",
    "    # 주중 서브플롯 (1행 2열 중 두 번째)\n",
    "    plt.subplot(1, 2, 2)\n",
    "    plt.plot(weekday_df[(weekday_df['month'] == 6)].groupby('hour')['power_consumption'].mean(), label='weekday6', color='orange')\n",
    "    plt.plot(weekday_df[(weekday_df['month'] == 7)].groupby('hour')['power_consumption'].mean(), label='weekday7', color='purple')\n",
    "    plt.plot(weekday_df[(weekday_df['month'] == 8)].groupby('hour')['power_consumption'].mean(), label='weekday8', color='cyan')\n",
    "    plt.legend()\n",
    "    plt.xticks(list(range(24)))\n",
    "    plt.ylim(0, 3700)  # y축 최대값 설정\n",
    "    plt.title('Weekday Consumption per 24 hours')\n",
    "\n",
    "    plt.tight_layout()  # 서브플롯 간 간격 조절\n",
    "    return plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3eaa192",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36e351b3",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "724d3802",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "jaypark",
   "language": "python",
   "name": "jaypark"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
