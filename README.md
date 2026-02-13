# federated-learning-1dcnn
IMUセンサデータを用いた連合学習実験コード

## 概要

- **目的**: IMUデータから特定のパターンや状態を分類
- **手法**: 1D CNN（畳み込みニューラルネットワーク）
- **データ形式**: 3軸加速度・角速度データ（.npy形式）
- **分類**: 二値分類（positive/negative）

## プロジェクト構成

```
fl_imu/
├── server.py                   #Flowerサーバ
├── client.py                   #Flowerクライアント
├── train_1dcnn.py              #モデル定義 & 学習関数
├── split_npy_to_clients.py     #データ分配
├── computeg0g1.py              #勾配作成関数
├── simulationFedAvg.py         #FedAvg
├── simulationgradFedRE.py      #gradFL
├── neg                         #負例データ
├── pos                         #正例データ
├── data/                       # 学習データ
│    ├──client1/                
│    │    ├── pos/　　　　　　　　　#正例データ
│    │    │   ├── imu_1.npy
│    │    │   └── imu_2.npy
│    │    └── neg/　　　　　　　　　#負例データ
│    │        ├── imu_3.npy
│    │        └── imu_4.npy
│    ├──client2/ 
```

## 必要なパッケージ
以下のパッケージを先にインストールする
```bash
!pip install -U "flwr[simulation]"
!pip install torch torchvision torchaudio
!pip install numpy
```

## データ形式

- **入力**: 3軸IMUデータ（加速度・角速度）
- **形状**: `(3, N)` - 3軸 × 時系列長
- **正規化**: Z-score正規化
- **パディング**: パディング（N < 200 の場合に 200 に揃える）

## モデル構成

### CNNBinaryClassifier
- **モデル名**: CNNBinaryClassifier (1D CNN)
- **入力チャンネル**: 3（加速度・角速度の3軸）
- **ベースチャンネル**: 32
- **構造**:
  - conv1: Conv1d, 入力チャネル=3, 出力チャネル=16, kernel=5, padding=2
  - bn1: BatchNorm1d(16)
  - conv2: Conv1d, 入力チャネル=16, 出力チャネル=32, kernel=5, padding=2
  - bn2: BatchNorm1d(32)
  - fc1: Linear(32 * 200 → 64)
  - fc2: Linear(64 → 2)
- **活性化関数**: ReLU
- **Flatten**: 全結合層の前に1次元に展開
- **出力**: 2クラス（二値分類）


## 使用方法

### 1. データの配分
以下のコードを実行、クライアント数とposデータ、negデータのパスを指定

dateディレクトリ以下にランダムかつ均等に分配される
```bash
%cd fl_imu_projectgradRE
!python split_npy_to_clients.py
```

### 2. proxyデータを用いた勾配作成
%cd fl_imu_projectgradRE
!python computeg0g1.py
を実行するとg0.npy/g1.npy が保存される



### 3. モデル学習の準備
simulationgradFedRE.pyの
G0_PATH = "g0_fc2weight.npy"
G1_PATH = "g1_fc2weight.npy"
を2で作成したg0.npy/g1.npy のパスに書き換える
またlast_layer_index=17にする

### 4. モデル学習の実行
simulationFedAvg.py、simulationgradFedRE.pyの
NUM_CLIENTS、NUM_ROUNDS、BATCH_SIZE、LOCAL_EPOCHSを設定する
(例：NUM_CLIENTS = 10、NUM_ROUNDS = 15
BATCH_SIZE = 16、LOCAL_EPOCHS = 2)

%cd fl_imu_projectgradRE
!python simulationFedAvg.py
で通常の連合学習の学習を実行

%cd fl_imu_projectgradRE
!python simulationgradFedRE.py
で重み付きありの連合学習を実行


## データ前処理
- 正規化: Z-score 各軸ごと
- パディング: 200 時系列長に揃える
- バッチ作成: DataLoader で shuffle=True（学習）、shuffle=False（検証）
- ラベル: pos → 1, neg → 0

## 学習設定

- **エポック数**: 2
- **学習ラウンド**：15（サーバー側 num_rounds） 
- **バッチサイズ**: 16（訓練）/ 16（検証）
- **最適化器**: AdamW (lr=1e-3)
- **損失関数**: CrossEntropyLoss  


## 注意事項
