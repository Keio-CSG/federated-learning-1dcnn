# -----------------------------
# Flower サーバー起動コード（FedAvg）
# -----------------------------

import flwr as fl

# Accuracy を平均する関数
def weighted_average(metrics):
    # metrics: List[Tuple[num_examples, Dict[str, float]]]
    accuracies = []
    total_examples = 0
    for num_examples, metric in metrics:
        accuracies.append(metric["accuracy"] * num_examples)
        total_examples += num_examples
    return {"accuracy": sum(accuracies) / total_examples}

# -----------------------------
# サーバーの戦略（Federated Averaging）設定
# -----------------------------
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=9,       # 1ラウンドで学習に参加する最小クライアント数
    min_available_clients=9, # サーバー起動時に必要なクライアント数
    evaluate_metrics_aggregation_fn=weighted_average
)

# -----------------------------
# サーバー起動
# -----------------------------
fl.server.start_server(
    server_address="localhost:8080",  # サーバーのIP:PORT（ローカルテスト用）
    strategy=strategy,                # FedAvg 戦略を使用
    config=fl.server.ServerConfig(
        num_rounds=15                 # 全ラウンド数
    )
)

# -----------------------------
# 補足
# -----------------------------
# 1. サーバーは必ずクライアントより先に起動すること
# 2. クライアントは client.py などで server_address="localhost:8080" に接続
# 3. min_fit_clients や min_available_clients はクライアント数に合わせて調整可能
# 4. 将来的には start_server() は非推奨で、flower-superlink CLI の利用が推奨されます
