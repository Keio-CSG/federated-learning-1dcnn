# -----------------------------
# Flower クライアント (PyTorch 1D CNN) サンプル
# -----------------------------

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_1dcnn import CNNBinaryClassifier, NpyWaveDataset
from sklearn.model_selection import train_test_split
import numpy as np
import sys

# -----------------------------
# 実行時引数からクライアントIDを取得
# 例: python client.py client1
# -----------------------------
client_id = sys.argv[1]
data_dir = f"./data/{client_id}"

# -----------------------------
# デバイス設定（GPUがあればGPUを使用）
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# モデル作成
# -----------------------------
model = CNNBinaryClassifier().to(device)

# -----------------------------
# データ読み込み
# -----------------------------
dataset = NpyWaveDataset(data_dir)

# stratify でクラス比を保った分割
labels = [dataset[i][1].item() for i in range(len(dataset))]
train_idx, val_idx = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)


train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=8, shuffle=False)

# -----------------------------
# 損失関数と最適化手法
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 1エポック分の学習関数
# -----------------------------
def train_one_epoch():
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# -----------------------------
# 検証関数
# -----------------------------
def evaluate():
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * len(y)
            correct += (preds.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total

# -----------------------------
# Flower クライアントクラス定義
# -----------------------------
class FLClient(fl.client.NumPyClient):
    # モデルの重みを返す
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    # サーバーから受け取った重みをモデルにセット
    def set_parameters(self, params):
        state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)}
        model.load_state_dict(state_dict, strict=True)

    # ローカル学習
    def fit(self, params, config):
        self.set_parameters(params)
        local_epochs = 1  # ラウンドごとのローカル学習エポック数
        for _ in range(local_epochs):
            train_one_epoch()
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    # 検証
    def evaluate(self, params, config):
        self.set_parameters(params)
        val_loss, val_acc = evaluate()
        return float(val_loss), len(val_loader.dataset), {"accuracy": float(val_acc)}

# -----------------------------
# Flower クライアント起動（最新仕様に対応）
# -----------------------------
# NumPyClient を Client オブジェクトに変換してから起動
client = FLClient().to_client()

fl.client.start_client(
    server_address="localhost:8080",  # サーバーと同じアドレスに接続
    client=client
)

# -----------------------------
# 注意点
# -----------------------------
# 1. サーバーを先に起動してからクライアントを起動すること
# 2. local_epochs や batch_size はデータセットや学習速度に応じて調整
# 3. 非推奨 warning を回避でき、将来的にも対応可能

