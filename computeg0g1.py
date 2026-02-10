import os
import numpy as np
import torch

from train_1dcnn import CNNBinaryClassifier, NpyWaveDataset


# ==============================
# 設定
# ==============================
PROXY_DATA_DIR = "./data/proxy"  # 両クラス(0/1)が混ざった小さなデータセットを置く
BATCH_SIZE_PER_CLASS = 32        # 各クラスから何サンプル使うか（足りなければあるだけ使う）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_class_batches(dataset, batch_size_per_class=32):
    """
    NpyWaveDataset からクラス0とクラス1のバッチをそれぞれ作る。
    - バッチサイズに足りない場合は、あるだけ使う。
    """
    xs0, ys0 = [], []
    xs1, ys1 = [], []

    for x, y in dataset:
        if y == 0 and len(xs0) < batch_size_per_class:
            xs0.append(x)
            ys0.append(y)
        elif y == 1 and len(xs1) < batch_size_per_class:
            xs1.append(x)
            ys1.append(y)

        if len(xs0) >= batch_size_per_class and len(xs1) >= batch_size_per_class:
            break

    if len(xs0) == 0 or len(xs1) == 0:
        raise RuntimeError(
            f"proxy データ {PROXY_DATA_DIR} にクラス0/1の両方が十分に入っているか確認してください。"
        )

    x0 = torch.stack(xs0, dim=0)
    y0 = torch.tensor(ys0, dtype=torch.long)
    x1 = torch.stack(xs1, dim=0)
    y1 = torch.tensor(ys1, dtype=torch.long)

    return (x0, y0), (x1, y1)


def get_target_param_info(model):
    """
    model.state_dict() のうち「学習可能なパラメータ」に対応する key を列挙し、
    最後のものを「ターゲット」として選ぶ。

    返り値:
      target_key: state_dict 上のキー（例: 'fc2.bias'）
      target_index: state_dict.keys() の中での index（Flower の last_layer_index 用）
    """
    sd = model.state_dict()
    sd_keys = list(sd.keys())  # Flower 側もこの順番で parameters_to_ndarrays を並べるはず

    # named_parameters() から「学習可能パラメータ」の dict を作る
    named_params = dict(model.named_parameters())

    # state_dict の順番の中で、「学習可能パラメータに対応する key」だけ抜き出す
    trainable_keys = [k for k in sd_keys if k in named_params]

    if not trainable_keys:
        raise RuntimeError("学習可能なパラメータが見つかりませんでした。CNNBinaryClassifier を確認してください。")

    # 一番最後の学習可能パラメータをターゲットにする（たいてい最終層の weight/bias）
    target_key = trainable_keys[-1]
    target_index = sd_keys.index(target_key)  # state_dict 全体の中でのインデックス

    print("==== state_dict のキー一覧（index: name (shape)）====")
    for idx, k in enumerate(sd_keys):
        print(f"{idx:3d}: {k:30s} {tuple(sd[k].shape)}")
    print("====================================================")
    print(f"選ばれたターゲットパラメータ: '{target_key}' (index = {target_index})")
    print("→ GradFedREStrategy(last_layer_index=この index) と合わせてください。")

    return target_key, target_index


def compute_gradient_for_class(model, criterion, batch, target_key):
    """
    あるクラス（0 or 1）のバッチに対して、
    target_key に対応するパラメータの勾配ベクトルを計算して flatten して返す。
    """
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    model.zero_grad()
    model.train()

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # 勾配を取りたいパラメータ tensor
    param_dict = dict(model.named_parameters())  # name -> Parameter
    if target_key not in param_dict:
        raise RuntimeError(f"target_key '{target_key}' が named_parameters に見つかりません。")

    target_param = param_dict[target_key]

    if target_param.grad is None:
        raise RuntimeError(f"パラメータ '{target_key}' に grad が付きませんでした。計算グラフを確認してください。")

    grad_vec = target_param.grad.detach().cpu().flatten().numpy()
    return grad_vec


def main():
    # 1. プロキシデータ読み込み
    if not os.path.exists(PROXY_DATA_DIR):
        raise RuntimeError(f"PROXY_DATA_DIR が存在しません: {PROXY_DATA_DIR}")

    dataset = NpyWaveDataset(PROXY_DATA_DIR)
    if len(dataset) == 0:
        raise RuntimeError(f"proxy データが空です: {PROXY_DATA_DIR}")

    (x0, y0), (x1, y1) = build_class_batches(dataset, BATCH_SIZE_PER_CLASS)
    print(f"クラス0バッチ: {x0.shape}, クラス1バッチ: {x1.shape}")

    # 2. モデル・損失関数の準備
    model = CNNBinaryClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. どのパラメータを g0/g1 用のターゲットにするか決める
    target_key, target_index = get_target_param_info(model)

    # 初期パラメータを保存しておく（g0,g1 を同じ初期値から計算するため）
    base_state = model.state_dict()

    # 4. クラス0用の勾配ベクトル g0
    g0 = compute_gradient_for_class(model, criterion, (x0, y0), target_key)
    print(f"g0 shape: {g0.shape}")

    # 5. クラス1用の勾配ベクトル g1
    #    ※ 同じ初期パラメータから計算したいので、state_dict を巻き戻す
    model.load_state_dict(base_state)
    g1 = compute_gradient_for_class(model, criterion, (x1, y1), target_key)
    print(f"g1 shape: {g1.shape}")

    # 6. 正規化（単位ベクトルにしておくと後で扱いやすい）
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    g0_norm = normalize(g0)
    g1_norm = normalize(g1)

    # 7. 保存
    np.save("g0.npy", g0_norm)
    np.save("g1.npy", g1_norm)
    print("g0.npy / g1.npy を保存しました。")
    print(f"このときの target_index = {target_index} を GradFedREStrategy(last_layer_index=...) に設定してください。")


if __name__ == "__main__":
    main()
