import os
import numpy as np
import flwr as fl
import torch
from torch.utils.data import DataLoader
from train_1dcnn import CNNBinaryClassifier, NpyWaveDataset
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# -----------------------------
# 設定
# -----------------------------
NUM_CLIENTS = 8
NUM_ROUNDS = 15
BATCH_SIZE = 16
LOCAL_EPOCHS = 2


def client_fn(cid: str):
    # Flower は "0", "1", ... を渡してくるので +1 して "1", "2", ... に合わせる
    cid_int = int(cid) + 1
    data_dir = f"./data/client{cid_int}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # データがない場合はダミークライアントを返す
    # =========================
    if (not os.path.exists(data_dir)) or (len(os.listdir(data_dir)) == 0):
        print(f"[Warning] Client {cid_int}: dataset is missing or empty ({data_dir}). Using DummyClient.")

        class DummyClient(fl.client.NumPyClient):
            def __init__(self):
                self.model = CNNBinaryClassifier().to(device)

            def get_parameters(self, config):
                return [v.cpu().numpy() for v in self.model.state_dict().values()]

            def set_parameters(self, parameters):
                state_dict = {
                    k: torch.tensor(v)
                    for k, v in zip(self.model.state_dict().keys(), parameters)
                }
                self.model.load_state_dict(state_dict, strict=True)

            def fit(self, parameters, config):
                # 何も学習せず、そのままパラメータを返す
                self.set_parameters(parameters)
                return self.get_parameters(config={}), 0, {}

            def evaluate(self, parameters, config):
                # サンプル数 0 として扱う（カウントも全部 0）
                self.set_parameters(parameters)
                return 0.0, 0, {
                    "correct": 0,
                    "minority_correct": 0,
                    "minority_total": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }

        return DummyClient()

    # =========================
    # ここから「実データありクライアント」
    # =========================
    dataset = NpyWaveDataset(data_dir)

    if len(dataset) == 0:
        print(f"[Warning] Client {cid_int}: dataset object is empty. Using DummyClient.")

        class DummyClient(fl.client.NumPyClient):
            def __init__(self):
                self.model = CNNBinaryClassifier().to(device)

            def get_parameters(self, config):
                return [v.cpu().numpy() for v in self.model.state_dict().values()]

            def set_parameters(self, parameters):
                state_dict = {
                    k: torch.tensor(v)
                    for k, v in zip(self.model.state_dict().keys(), parameters)
                }
                self.model.load_state_dict(state_dict, strict=True)

            def fit(self, parameters, config):
                self.set_parameters(parameters)
                return self.get_parameters(config={}), 0, {}

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                return 0.0, 0, {
                    "correct": 0,
                    "minority_correct": 0,
                    "minority_total": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                }

        return DummyClient()

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル・最適化
    model = CNNBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ローカルのクラス不均衡に対する重み付き Loss（これはそのまま利用）
    labels = np.array([y for _, y in dataset])
    counts = np.bincount(labels, minlength=2)
    counts_safe = np.maximum(counts, 1)
    total = len(labels)
    class_weights = torch.tensor(total / (2 * counts_safe), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    class FedREClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for val in model.state_dict().values()]

        def set_parameters(self, parameters):
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            # グローバルモデルからスタート
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
            # グローバルモデル（サーバから渡されたパラメータ）で評価
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

                    # 全サンプル数・正解数
                    correct += (pred_labels == y).sum().item()
                    total_samples += len(y)

                    # 少数クラス（1）のカウント
                    minority_mask = (y == 1)
                    minority_total += minority_mask.sum().item()
                    minority_correct += ((pred_labels == 1) & (y == 1)).sum().item()

                    # TP/FP/FN（クラス1を陽性として扱う）
                    TP += ((pred_labels == 1) & (y == 1)).sum().item()
                    FP += ((pred_labels == 1) & (y == 0)).sum().item()
                    FN += ((pred_labels == 0) & (y == 1)).sum().item()

            return 0.0, total_samples, {
                "correct": correct,
                "minority_correct": minority_correct,
                "minority_total": minority_total,
                "tp": TP,
                "fp": FP,
                "fn": FN,
            }

    return FedREClient()


# -----------------------------
# グローバル評価用：全クライアントの検証データをまとめた形で accuracy / F1 を計算
# -----------------------------
def global_metrics(metrics):
    total_examples = 0
    total_correct = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_minority_correct = 0
    total_minority_total = 0

    for num_examples, metric in metrics:
        total_examples += num_examples
        total_correct += metric.get("correct", 0)

        total_tp += metric.get("tp", 0)
        total_fp += metric.get("fp", 0)
        total_fn += metric.get("fn", 0)

        total_minority_correct += metric.get("minority_correct", 0)
        total_minority_total += metric.get("minority_total", 0)

    print(
        "[global_metrics] "
        f"examples={total_examples}, "
        f"tp={total_tp}, fp={total_fp}, fn={total_fn}, "
        f"minority_total={total_minority_total}"
    )

    if total_examples == 0:
        return {"accuracy": 0.0, "minority_accuracy": 0.0, "f1": 0.0}

    # グローバル accuracy（全正解数 / 全サンプル数）
    accuracy = total_correct / total_examples

    # グローバル minority_accuracy（クラス1だけの accuracy）
    if total_minority_total > 0:
        minority_accuracy = total_minority_correct / total_minority_total
    else:
        minority_accuracy = 0.0

    # グローバル precision / recall / F1（クラス1を陽性とする）
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0.0

    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": accuracy,
        "minority_accuracy": minority_accuracy,
        "f1": f1,
    }


# -----------------------------
# 勾配ベース FedRE 風の Strategy
# -----------------------------
class GradFedREStrategy(fl.server.strategy.FedAvg):
    def __init__(self, g0, g1, last_layer_index=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # サーバーが持っている「基底勾配」（単位ベクトルに正規化）
        self.g0 = g0 / np.linalg.norm(g0)
        self.g1 = g1 / np.linalg.norm(g1)
        self.last_layer_index = last_layer_index
        # 前ラウンドのグローバルパラメータ（ndarray リスト）
        self.prev_global_params = None

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # 初回は普通の FedAvg で集約して、その結果を prev_global_params に保存
        if self.prev_global_params is None:
            out = super().aggregate_fit(server_round, results, failures)
            if out[0] is not None:
                self.prev_global_params = parameters_to_ndarrays(out[0])
            return out

        weighted_params = None
        total_weight = 0.0
        eps = 1e-8

        for client, fit_res in results:
            params_nd = parameters_to_ndarrays(fit_res.parameters)
            n_i = fit_res.num_examples

            # --- 最終層の更新ベクトル Δw_k ---
            w_global_last = self.prev_global_params[self.last_layer_index].ravel()
            w_k_last = params_nd[self.last_layer_index].ravel()
            delta_w = w_k_last - w_global_last

            # g0/g1 と次元を揃える（短い方に合わせる）
            dim = min(len(delta_w), len(self.g0), len(self.g1))
            dw = delta_w[:dim]
            g0 = self.g0[:dim]
            g1 = self.g1[:dim]

            # --- 勾配基底への射影からクラス比率を推定 ---
            alpha0 = max(float(np.dot(dw, g0)), 0.0)
            alpha1 = max(float(np.dot(dw, g1)), 0.0)
            s = alpha0 + alpha1

            # ほとんど情報がないときは中立（0.5）にする
            if s < 1e-3:
                minority_ratio = 0.5
            else:
                minority_ratio = alpha1 / (s + eps)

            # 極端な値を防ぐためにクリップ
            minority_ratio = float(np.clip(minority_ratio, 0.1, 0.9))

            # FedAvg をベースに少しだけ傾ける重み
            alpha_factor = 1.0  # 0 にすると純粋な FedAvg
            center = 0.5
            bias = minority_ratio - center  # -0.4〜+0.4 くらい
            scale = 1.0 + alpha_factor * bias

            # どのクライアントも完全には無視しないように下限を設定
            min_scale = 0.1
            scale = max(scale, min_scale)

            w_i = n_i * scale
            if w_i <= 0:
                continue

            print(
                f"[round {server_round}] client={getattr(client, 'cid', '?')}, "
                f"n_i={n_i}, alpha0={alpha0:.4f}, alpha1={alpha1:.4f}, "
                f"minority_ratio={minority_ratio:.4f}, scale={scale:.3f}, w_i={w_i:.2f}"
            )

            # --- 重み付き和を計算 ---
            if weighted_params is None:
                weighted_params = [w_i * p for p in params_nd]
            else:
                weighted_params = [
                    wp + w_i * p for wp, p in zip(weighted_params, params_nd)
                ]
            total_weight += w_i

        # もし推定に失敗したら通常の FedAvg にフォールバック
        if weighted_params is None or total_weight == 0.0:
            out = super().aggregate_fit(server_round, results, failures)
            if out[0] is not None:
                self.prev_global_params = parameters_to_ndarrays(out[0])
            return out

        new_params = [p / total_weight for p in weighted_params]
        # 次ラウンドの Δw 計算のために保存
        self.prev_global_params = [p.copy() for p in new_params]

        return ndarrays_to_parameters(new_params), {}


# -----------------------------
# g0, g1 の用意
# -----------------------------
g0 = np.load("g0.npy")
g1 = np.load("g1.npy")

strategy = GradFedREStrategy(
    g0=g0,
    g1=g1,
    last_layer_index=17,  # 今は fc2.bias 用（g0/g1 を fc2.weight に変えたら 16 に変更）
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=global_metrics,
)

# -----------------------------
# シミュレーション開始
# -----------------------------
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
