import os
import flwr as fl
import torch
from torch.utils.data import DataLoader
from train_1dcnn import CNNBinaryClassifier, NpyWaveDataset
import numpy as np

# -----------------------------
# 設定
# -----------------------------
NUM_CLIENTS = 8
NUM_ROUNDS = 15
BATCH_SIZE = 16
LOCAL_EPOCHS = 2

# -----------------------------
# クライアント作成関数
# -----------------------------
def client_fn(cid: str):
    # Flowerは "0", "1" を渡してくるので +1 して "1", "2" にする
    cid_int = int(cid) + 1
    data_dir = f"./data/client{cid_int}"

    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        raise RuntimeError(f"Client {cid_int} dataset is empty or missing: {data_dir}")

    dataset = NpyWaveDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBinaryClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    class SimClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for val in model.state_dict().values()]

        def set_parameters(self, parameters):
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            for _ in range(LOCAL_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
            return self.get_parameters(config={}), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            model.eval()

            total_samples, correct = 0, 0
            minority_total, minority_correct = 0, 0
            TP = FP = FN = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x)
                    pred_labels = preds.argmax(1)

                    # 全体精度
                    correct += (pred_labels == y).sum().item()
                    total_samples += len(y)

                    # 少数クラス（1）の精度
                    minority_mask = (y == 1)
                    minority_total += minority_mask.sum().item()
                    minority_correct += ((pred_labels == 1) & (y == 1)).sum().item()

                    # 混同行列（クラス1を陽性として扱う）
                    TP += ((pred_labels == 1) & (y == 1)).sum().item()
                    FP += ((pred_labels == 1) & (y == 0)).sum().item()
                    FN += ((pred_labels == 0) & (y == 1)).sum().item()

            overall_acc = correct / total_samples

            if minority_total > 0:
                minority_acc = minority_correct / minority_total
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                minority_acc = 0.0
                f1 = 0.0

            minority_ratio = minority_total / total_samples if total_samples > 0 else 0.0

            return 0.0, total_samples, {
                "accuracy": overall_acc,
                "minority_accuracy": minority_acc,
                "f1": f1,
                "minority_ratio": minority_ratio,
            }

    return SimClient()



# -----------------------------
# サーバ側の評価集約（F1追加）
# -----------------------------
def weighted_average(metrics):
    acc_sum = 0
    minority_sum = 0
    f1_sum = 0
    total_examples = 0

    for num_examples, metric in metrics:
        acc_sum += metric["accuracy"] * num_examples
        minority_sum += metric["minority_accuracy"] * num_examples
        f1_sum += metric["f1"] * num_examples
        total_examples += num_examples

    return {
        "accuracy": acc_sum / total_examples,
        "minority_accuracy": minority_sum / total_examples,
        "f1": f1_sum / total_examples
    }

# -----------------------------
# サーバ戦略
# -----------------------------
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average
)

# -----------------------------
# シミュレーション開始
# -----------------------------
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy
)