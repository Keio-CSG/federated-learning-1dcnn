import os
import shutil
import random
from pathlib import Path
import numpy as np

def distribute_data(src_pos, src_neg, num_clients, mode="iid", base_dir="./data"):
    src_pos = Path(src_pos)
    src_neg = Path(src_neg)
    base = Path(base_dir)

    # --- load data files ---
    pos_files = list(src_pos.glob("*.npy"))
    neg_files = list(src_neg.glob("*.npy"))
    random.shuffle(pos_files)
    random.shuffle(neg_files)

    print(f"pos ファイル数: {len(pos_files)}")
    print(f"neg ファイル数: {len(neg_files)}")

    if len(pos_files) == 0 or len(neg_files) == 0:
        print("❌ pos または neg に .npy がありません")
        return

    # --- Remove old client folders ---
    if base.exists():
        for d in base.glob("client*"):
            shutil.rmtree(d)
        print("🧹 既存のクライアントフォルダを削除しました。")

    # --- Create new client folders ---
    for i in range(1, num_clients + 1):
        (base / f"client{i}" / "pos").mkdir(parents=True, exist_ok=True)
        (base / f"client{i}" / "neg").mkdir(parents=True, exist_ok=True)

    # --- Assign data to each client ---
    if mode == "iid":
        print("IID モード: pos/neg を均等に分配")
        total_pos = len(pos_files)
        total_neg = len(neg_files)
        pos_per_client = total_pos // num_clients
        neg_per_client = total_neg // num_clients

        for cid in range(1, num_clients + 1):
            client_pos_dir = base / f"client{cid}" / "pos"
            client_neg_dir = base / f"client{cid}" / "neg"

            selected_pos = pos_files[:pos_per_client]
            selected_neg = neg_files[:neg_per_client]

            for f in selected_pos:
                shutil.copy(f, client_pos_dir)
            for f in selected_neg:
                shutil.copy(f, client_neg_dir)

            pos_files = pos_files[pos_per_client:]
            neg_files = neg_files[neg_per_client:]

            print(f"client{cid}: pos={len(selected_pos)}, neg={len(selected_neg)}")

    else:  # Non-IID
        print("Non-IID モード: pos/neg 区別せずランダムに分配")
        all_files = pos_files + neg_files
        random.shuffle(all_files)
        per_client = len(all_files) // num_clients

        for cid in range(1, num_clients + 1):
            client_pos_dir = base / f"client{cid}" / "pos"
            client_neg_dir = base / f"client{cid}" / "neg"

            start_idx = (cid - 1) * per_client
            end_idx = start_idx + per_client
            client_files = all_files[start_idx:end_idx]

            pos_count = 0
            neg_count = 0
            for f in client_files:
                if "pos" in str(f):
                    shutil.copy(f, client_pos_dir)
                    pos_count += 1
                else:
                    shutil.copy(f, client_neg_dir)
                    neg_count += 1

            print(f"client{cid}: pos={pos_count}, neg={neg_count}")

        # 余りは最後のクライアントに追加
        remaining_files = all_files[per_client * num_clients:]
        for f in remaining_files:
            if "pos" in str(f):
                shutil.copy(f, base / f"client{num_clients}" / "pos")
            else:
                shutil.copy(f, base / f"client{num_clients}" / "neg")

    print("\n🎉 完了：データ配分が破綻しない Non-IID / IID 分配が完了しました！")


if __name__ == "__main__":
    num_clients = int(input("クライアント数: "))
    src_pos = input("元の pos ディレクトリ: ").strip()
    src_neg = input("元の neg ディレクトリ: ").strip()
    mode = input("モードを選択 (iid / non-iid): ").strip().lower()

    distribute_data(src_pos, src_neg, num_clients, mode)
