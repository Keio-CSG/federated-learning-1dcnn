# federated-learning-1dcnn
IMUセンサデータを用いた連合学習実験コード

## 概要
- **目的**: IMUデータを用いた二値分類の連合学習実験コード 
- **手法**: 1D CNN（畳み込みニューラルネットワーク）  
- **データ形式**: 3軸加速度・角速度データ（.npy形式）  
- **分類**: 二値分類（positive / negative）

  
## プロジェクト構成
fl_imu_project/
├── server.py                   # Flowerサーバ
├── client.py                   #クライアント
├── train_1dcnn.py              #モデル定義 & 学習関数
├── split_npy_to_clients.py     #データ分配
└── data/
　　　├── client1/
　　　│   　　├── pos/　　　　　　　　　#正例データ
　　　│   　　│   　　├── imu_1.npy
　　　│　   　│   　　└── imu_2.npy
　　　│   　　└── neg/　　　　　　　　　#負例データ
　　　│       　　　　├── imu_3.npy
　　　│       　　　　└── imu_4.npy
　　　├── client2/
　　　│  　　 ├── pos/
　　　│   　　└── neg/
　　　└── client3/
　　　        ├── pos/
　　          └── neg/


## 必要なパッケージ
pip install torch torchvision          
pip install numpy                      
pip install scikit-learn              
pip install flwr                       
pip install matplotlib                 
pip install rosbags opencv-python      

## データ形式
・データファイル: .npy
・形状: (3, N) → 3軸 × 時系列長
・内容: 加速度・角速度など IMU データ
・前処理:
・Z-score 正規化（各軸ごと）
・パディング（N < 200 の場合に 200 に揃える）
・分割: 学習・検証は train_test_split で stratify して二値ラベル比を維持

## 使用方法
1.データ分配　python split_npy_to_clients.py
- 正例データと負例データを分けて配置しクライアント数と配置したパスを指定
dateディレクトリ以下にランダムかつ均等に分配される
2.サーバ起動 python server.py
3.クライアント起動 python client.py client1(ファイル名)
- ローカルエポック数 client.py 内で変更
- ラウンド数、クライアント数server.py内で変更
- サーバー、各クライアントそれぞれ別のターミナルで実行

## データ前処理
・正規化: Z-score 各軸ごと
・パディング: 200 時系列長に揃える
・バッチ作成: DataLoader で shuffle=True（学習）、shuffle=False（検証）
・ラベル: pos → 1, neg → 0

## 学習設定
・損失関数：CrossEntropyLoss     
・最適化器：Adam (lr=1e-3)       
・バッチサイズ：16                   
・ローカル学習エポック：2（Flowerクライアント毎ラウンド）
・学習ラウンド：15（サーバー側 num_rounds） 
