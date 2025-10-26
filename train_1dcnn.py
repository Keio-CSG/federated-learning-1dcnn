import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -----------------------------
# 1D CNN モデル
# -----------------------------
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32 * 200, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# Dataset定義
# -----------------------------
class NpyWaveDataset(Dataset):
    def __init__(self, data_dir):
        pos_dir = f"{data_dir}/pos"
        neg_dir = f"{data_dir}/neg"

        self.data = []
        self.labels = []

        for npy in list(Path(pos_dir).glob("*.npy")):
            self.data.append(np.load(npy))
            self.labels.append(1)
        for npy in list(Path(neg_dir).glob("*.npy")):
            self.data.append(np.load(npy))
            self.labels.append(0)

        # Z-score正規化
        self.data = [ (d - np.mean(d, axis=1, keepdims=True)) / (np.std(d, axis=1, keepdims=True) + 1e-8)
                      for d in self.data ]

        # パディング（N < 200対応）
        self.data = [ np.pad(d, ((0,0), (0, max(0, 200 - d.shape[1]))))[:, :200] for d in self.data ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
