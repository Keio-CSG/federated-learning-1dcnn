import flwr as fl
import torch
from torch.utils.data import DataLoader
from train_1dcnn import CNNBinaryClassifier, NpyWaveDataset
import numpy as np

# -----------------------------
# 設定
# -----------------------------
NUM_CLIENTS = 18
NUM_ROUNDS = 15
BATCH_SIZE = 16
LOCAL_EPOCHS = 2

# -----------------------------
# クライアント作成関数
# -----------------------------
def client_fn(cid: str):
    # データロード
    data_dir = f"./data/client{cid}"
    dataset = NpyWaveDataset(data_dir)

    if len(dataset) == 0:
        raise ValueError(f"Client {cid} dataset is empty!")

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル
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
            total, correct = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x)
                    correct += (preds.argmax(1) == y).sum().item()
                    total += len(y)
            return 0.0, total, {"accuracy": correct / total}

    return SimClient()

# -----------------------------
# サーバ戦略
# -----------------------------
def weighted_average(metrics):
    accuracies = []
    total_examples = 0
    for num_examples, metric in metrics:
        accuracies.append(metric["accuracy"] * num_examples)
        total_examples += num_examples
    return {"accuracy": sum(accuracies) / total_examples}

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
