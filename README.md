# ⚡ 2023 전력사용량 예측 AI 경진대회 

## 📋 목차

* [📝 대회 설명](#competition)
* [💾 데이터셋 설명](#dataset)
* [🗄 디렉토리 구조](#folder)
* [⚙️ 설정 사항](#setup)
* [💻 실행하는 법](#torun)
<br><br/>

---
<br><br>

## ⚙️ 설정 사항 <a name='setup'></a>

### 1. Conda Create
```bash
$ conda create -n pyto python=3.9

$ conda activate pyto
```

### 2. Pytorch Install (with CUDA.)
```bash
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 3. Requirements

```bash
$ pip install -r requirements.txt
```
<br><br>
## 💻 실행하는 법 <a name='torun'></a>

### Train

```bash
$ python main.py -m t
or
$ python main.py -m train
```

### Inference

```bash
$ python main.py -m i
or
$ python main.py -m inference
```