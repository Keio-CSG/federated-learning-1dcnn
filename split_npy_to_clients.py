import os
import shutil
import random
from pathlib import Path

def distribute_data(src_pos, src_neg, num_clients, base_dir="./data"):
    """
    pos/neg の .npy ファイルを指定したクライアント数に均等・ランダム分配。
    """
    src_pos = Path(src_pos)
    src_neg = Path(src_neg)
    base = Path(base_dir)

    # コピー元の確認
    pos_files = list(src_pos.glob("*.npy"))
    neg_files = list(src_neg.glob("*.npy"))

    print(f"検出された pos ファイル数: {len(pos_files)}")
    print(f"検出された neg ファイル数: {len(neg_files)}")

    if len(pos_files) == 0 or len(neg_files) == 0:
        print("❌ pos または neg フォルダに .npy ファイルが見つかりません。パスを確認してください。")
        print(f"posフォルダ: {src_pos.resolve()}")
        print(f"negフォルダ: {src_neg.resolve()}")
        return

    # 既存のクライアントフォルダを全削除
    if base.exists():
        for d in base.glob("client*"):
            shutil.rmtree(d)
        print("🧹 既存のクライアントフォルダを削除しました。")

    # pos/negファイルをランダムにシャッフル
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    # 各クライアントフォルダを作成
    for i in range(1, num_clients + 1):
        (base / f"client{i}" / "pos").mkdir(parents=True, exist_ok=True)
        (base / f"client{i}" / "neg").mkdir(parents=True, exist_ok=True)

    # 均等分配
    for files, label in [(pos_files, "pos"), (neg_files, "neg")]:
        for i, f in enumerate(files):
            client_id = (i % num_clients) + 1
            dst_dir = base / f"client{client_id}" / label
            shutil.copy(f, dst_dir)

    print(f"\n✅ {num_clients} クライアントにデータを均等・ランダムに分配しました。")

if __name__ == "__main__":
    num_clients = int(input("クライアント数を入力してください: "))
    src_pos = input("元のposディレクトリのパスを入力してください: ").strip()
    src_neg = input("元のnegディレクトリのパスを入力してください: ").strip()
    distribute_data(src_pos, src_neg, num_clients)

