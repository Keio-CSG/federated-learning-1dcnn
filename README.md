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
├── server.py                   # Flowerサーバ
├── client.py                   #クライアント
├── train_1dcnn.py              #モデル定義 & 学習関数
├── split_npy_to_clients.py     #データ分配
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

```bash
pip install torch torchvision          
pip install numpy                      
pip install scikit-learn              
pip install flwr                       
pip install matplotlib                 
pip install rosbags opencv-python
pip install -U "flwr[simulation]"
pip install torch torchvision torchaudio
pip install numpy     
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
正例データと負例データを分けて配置しコードを実行、クライアント数と配置したパスを指定

dateディレクトリ以下にランダムかつ均等に分配される
```bash
python split_npy_to_clients.py
```

### 2. サーバの起動

```bash
python server.py
```

### 3. クライアントの起動

```bash
python client.py client1(ファイル名)
```
- ローカルエポック数 client.py 内で変更
- ラウンド数、クライアント数server.py内で変更
- サーバー、各クライアントそれぞれ別のターミナルで実行

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
