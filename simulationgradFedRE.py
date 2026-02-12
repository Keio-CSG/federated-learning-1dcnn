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
NUM_CLIENTS = 10
NUM_ROUNDS = 15
BATCH_SIZE = 16
LOCAL_EPOCHS = 2

# ★ fc2.weight をターゲットにした g0/g1 を読む
G0_PATH = "g0_fc2weight.npy"
G1_PATH = "g1_fc2weight.npy"
LAST_LAYER_INDEX = 17  # ★ fc2.weight の index（state_dict 出力で確認した値に合わせる）

# サーバ側推定の温度（小さいほど推定が鋭くなる）
SOFTMAX_T = 0.2


def client_fn(cid: str):
    cid_int = int(cid) + 1
    data_dir = f"./data/client{cid_int}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # データがない場合はダミークライアント
    # =========================
    if (not os.path.exists(data_dir)) or (len(os.listdir(data_dir)) == 0):
        print(f"[Warning] Client {cid_int}: dataset is missing or empty ({data_dir}). Using DummyClient.")

        class DummyClient(fl.client.NumPyClient):
            def __init__(self):
                self.model = CNNBinaryClassifier().to(device)

            def get_parameters(self, config):
                return [v.cpu().numpy() for v in self.model.state_dict().values()]

            def set_parameters(self, parameters):
                state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
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

    # =========================
    # 実データありクライアント
    # =========================
    dataset = NpyWaveDataset(data_dir)
    if len(dataset) == 0:
        print(f"[Warning] Client {cid_int}: dataset object is empty. Using DummyClient.")
        return client_fn(str(cid_int - 1))

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNBinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ローカル不均衡への重み付きLoss（継続）
    labels = np.array([y for _, y in dataset])
    counts = np.bincount(labels, minlength=2)
    counts_safe = np.maximum(counts, 1)
    total = len(labels)
    class_weights = torch.tensor(total / (2 * counts_safe), dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ★ デバッグ用：真の minority 比率
    true_ratio = float(counts[1] / total) if total > 0 else 0.0

    class FedREClient(fl.client.NumPyClient):
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

            # ★ cid_int, true_ratio を返してサーバ側で検証できるようにする
            return self.get_parameters(config={}), len(train_loader.dataset), {
                "cid_int": cid_int,
                "true_ratio": true_ratio,
            }

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

                    correct += (pred_labels == y).sum().item()
                    total_samples += len(y)

                    minority_mask = (y == 1)
                    minority_total += minority_mask.sum().item()
                    minority_correct += ((pred_labels == 1) & (y == 1)).sum().item()

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
        f"examples={total_examples}, tp={total_tp}, fp={total_fp}, fn={total_fn}, "
        f"minority_total={total_minority_total}"
    )

    if total_examples == 0:
        return {"accuracy": 0.0, "minority_accuracy": 0.0, "f1": 0.0}

    accuracy = total_correct / total_examples
    minority_accuracy = (total_minority_correct / total_minority_total) if total_minority_total > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"accuracy": accuracy, "minority_accuracy": minority_accuracy, "f1": f1}


class GradFedREStrategy(fl.server.strategy.FedAvg):
    def __init__(self, g0, g1, last_layer_index=-1, softmax_t=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # g0/g1 はベクトルとして正規化
        self.g0 = g0 / (np.linalg.norm(g0) + 1e-12)
        self.g1 = g1 / (np.linalg.norm(g1) + 1e-12)
        self.last_layer_index = last_layer_index
        self.prev_global_params = None
        self.softmax_t = softmax_t

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # 初回はFedAvg
        if self.prev_global_params is None:
            out = super().aggregate_fit(server_round, results, failures)
            if out[0] is not None:
                self.prev_global_params = parameters_to_ndarrays(out[0])
            return out

        weighted_params = None
        total_weight = 0.0

        for client, fit_res in results:
            params_nd = parameters_to_ndarrays(fit_res.parameters)
            n_i = fit_res.num_examples

            # --- 最終層の更新ベクトル Δw ---
            w_global_last = self.prev_global_params[self.last_layer_index].ravel()
            w_k_last = params_nd[self.last_layer_index].ravel()
            delta_w = w_k_last - w_global_last

            # ★重要：更新は概ね「-勾配方向」なので、勾配ベクトル(g0/g1)に合わせるなら符号反転
            #     dw ≈ -grad
            dim = min(len(delta_w), len(self.g0), len(self.g1))
            dw = (-delta_w)[:dim]
            g0 = self.g0[:dim]
            g1 = self.g1[:dim]

            # --- cos類似度（安定のため） ---
            dw_norm = np.linalg.norm(dw) + 1e-12
            sim0 = float(np.dot(dw, g0) / dw_norm)
            sim1 = float(np.dot(dw, g1) / dw_norm)

            # --- softmax で minority_ratio を推定 ---
            T = self.softmax_t
            e0 = float(np.exp(sim0 / T))
            e1 = float(np.exp(sim1 / T))
            minority_ratio = float(e1 / (e0 + e1))

            # 端に張り付きすぎを防ぐ（検証中は 0.01〜0.99 くらいが便利）
            minority_ratio = float(np.clip(minority_ratio, 0.01, 0.99))

            # --- FedAvgベースに少しだけ傾ける ---
            alpha_factor = 1.0  # 0でFedAvg
            scale = 1.0 + alpha_factor * (minority_ratio - 0.5)
            scale = max(scale, 0.1)

            w_i = n_i * scale
            if w_i <= 0:
                continue

            # ログ（見やすいIDと真値）
            cid_int = fit_res.metrics.get("cid_int", "?")
            true_ratio = fit_res.metrics.get("true_ratio", None)
            print(
                f"[round {server_round}] client={cid_int}, n_i={n_i}, "
                f"sim0={sim0:.4f}, sim1={sim1:.4f}, "
                f"minority_ratio_hat={minority_ratio:.4f}, true_ratio={true_ratio}, "
                f"scale={scale:.3f}, w_i={w_i:.2f}"
            )

            if weighted_params is None:
                weighted_params = [w_i * p for p in params_nd]
            else:
                weighted_params = [wp + w_i * p for wp, p in zip(weighted_params, params_nd)]
            total_weight += w_i

        # 失敗したらFedAvg
        if weighted_params is None or total_weight == 0.0:
            out = super().aggregate_fit(server_round, results, failures)
            if out[0] is not None:
                self.prev_global_params = parameters_to_ndarrays(out[0])
            return out

        new_params = [p / total_weight for p in weighted_params]
        self.prev_global_params = [p.copy() for p in new_params]
        return ndarrays_to_parameters(new_params), {}


# -----------------------------
# g0, g1 読み込み（fc2.weight版）
# -----------------------------
g0 = np.load(G0_PATH)
g1 = np.load(G1_PATH)

if g0.shape != g1.shape:
    raise RuntimeError(f"g0.shape={g0.shape} と g1.shape={g1.shape} が一致しません。")
if g0.size != 128:
    print(f"[Warning] g0.size={g0.size}（128のはず）。fc2.weight版を読めているか確認してください。")

strategy = GradFedREStrategy(
    g0=g0,
    g1=g1,
    last_layer_index=LAST_LAYER_INDEX,  # ★ 16
    softmax_t=SOFTMAX_T,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=global_metrics,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

