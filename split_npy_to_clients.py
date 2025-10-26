import os
import shutil
import random
from pathlib import Path

def distribute_data(src_pos, src_neg, num_clients, base_dir="./data"):
    """
    pos/neg の .npy ファイルを、指定したクライアント数に均等かつランダムに分配する。
    既存のクライアントフォルダが存在する場合は削除してから分配。
    """
    pos_files = list(Path(src_pos).glob("*.npy"))
    neg_files = list(Path(src_neg).glob("*.npy"))

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    for i in range(1, num_clients + 1):
        client_pos = Path(base_dir) / f"client{i}" / "pos"
        client_neg = Path(base_dir) / f"client{i}" / "neg"

        # 既存フォルダを削除
        if client_pos.exists():
            shutil.rmtree(client_pos)
        if client_neg.exists():
            shutil.rmtree(client_neg)

        client_pos.mkdir(parents=True, exist_ok=True)
        client_neg.mkdir(parents=True, exist_ok=True)

    for files, label in [(pos_files, "pos"), (neg_files, "neg")]:
        for i, f in enumerate(files):
            client_id = (i % num_clients) + 1
            dst_dir = Path(base_dir) / f"client{client_id}" / label
            shutil.copy(f, dst_dir)
    
    print(f"\n✅ {num_clients} クライアントにデータをランダム・均等に分配しました。")

if __name__ == "__main__":
    num_clients = int(input("クライアント数を入力してください: "))
    src_pos = input("元のposディレクトリのパスを入力してください: ").strip()
    src_neg = input("元のnegディレクトリのパスを入力してください: ").strip()
    distribute_data(src_pos, src_neg, num_clients)
