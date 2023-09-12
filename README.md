# ⚡ Dolma팀 | 2023 전력사용량 예측 AI 경진대회

## 📋 목차

* [📝 대회 설명](#competition)
* [💾 데이터셋 설명](#dataset)
* [🗄 디렉토리 구조](#folder)
* [⚙️ 설정 사항](#setup)
* [💻 실행하는 법](#torun)
<br><br/>


## 📝 대회 설명 <a name='competition'></a>
<img src="asset/background.jpeg">

### 대회 성격
알고리즘 | 정형 | 시계열 | 에너지 | SMAPE <br>
2023.07.17 ~ 2023.08.28


### 대회 배경
* 안정적이고 효율적인 에너지 공급을 위해서는 전력 사용량에 대한 정확한 예측이 필요

* 따라서 한국에너지공단에서 건물과 시공간 정보를 활용하여 특정 시점의 전력 사용량을 예측하는 AI 모델 개발 대회를 개최 

<br><br>

## 💾 데이터셋 설명 <a name='dataset'></a>

1. train.csv
    * 100개 건물들의 2022년 06월 01일부터 2022년 08월 24일까지의 데이터
    * 일시별 기온, 강수량, 풍속, 습도, 일조, 일사 정보 포함
    * 전력사용량(kWh) 포함


2. building_info.csv
    * 100개 건물 정보
    * 건물 번호, 건물 유형, 연면적, 냉방 면적, 태양광 용량, ESS 저장 용량, PCS 용량


3. test.csv
    * 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 데이터
    * 일시별 기온, 강수량, 풍속, 습도의 예보 정보


4. sample_submission.csv
    * 대회 제출을 위한 양식
    * 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 전력사용량(kWh)을 예측
    * num_date_time은 건물번호와 시간으로 구성된 ID
    * 해당 ID에 맞춰 전력사용량 예측값을 answer 컬럼에 기입해야 함

<br><br>

## 🗄 디렉토리 구조 <a name='folder'></a>
```Plain Text
├──📁config
│   └── base.yaml
│
├──📁data
│   ├── building_info.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv → train_split.py 실행시) train_split.csv & valid.csv 생성
│
├──📁EDA
│   └── EDA_final.ipynb → EDA 노트북 파일
│
└──📁utils
│   ├── load.py
│   ├── loss_fn.py
│   ├── metrics.py
│   ├── preprocessing_fn.py
│   └── preprocessing.py
│
├── hp_search.py
├── inference.py
├── main.py
├── opt_search.py
├── README.md
├── requirements.txt
├── train_split.py
└── train.py
```

<br><br>

## ⚙️ 설정 사항 <a name='setup'></a>

### 1. Conda Create
```bash
$ conda create -n bigdata python=3.
$ conda activate bigdata
```

### 2. Requirements

```bash
$ pip install -r requirements.txt
```
<br><br>
## 💻 실행하는 법 <a name='torun'></a>

### Train

```bash
$ python main.py -m t
$ python main.py -m train
```

### Inference

```bash
$ python main.py -m i
$ python main.py -m inference
```

### Hyper-parameter Searching
```bash
$ python main.py -m h
$ python main.py -m hp_searching
```

### Optuna Searching
```bash
$ python main.py -m o
$ python main.py -m optuna
```