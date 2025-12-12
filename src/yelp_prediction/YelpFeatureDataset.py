import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

MAX_PHOTOS = 3
FEATURE_DIM = 1280  # EfficientNetV2-S output size


class YelpFeatureDataset(Dataset):
    def __init__(self, dataframe, features_dict: dict):
        self.features_dict = features_dict
        # Convert to dictionary for fast access
        self.data = dataframe.to_dicts()

        # Pre-allocate a zero vector for missing data
        self.zeros = torch.zeros(FEATURE_DIM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        photo_ids = row["photo_ids"]
        label = torch.tensor(row["stars"], dtype=torch.float32)

        # 1. SAMPLING Strategy suitable for MIL
        if len(photo_ids) == 0:
            # Edge case: No photos at all
            # Return a bag of zeros
            return torch.zeros(MAX_PHOTOS, FEATURE_DIM), label

        if len(photo_ids) >= MAX_PHOTOS:
            selected = np.random.choice(photo_ids, MAX_PHOTOS, replace=False)
        else:
            selected = np.random.choice(photo_ids, MAX_PHOTOS, replace=True)

        # 2. LOAD TENSORS
        features = []
        for pid in selected:
            # Fast in-memory lookup
            feat = self.features_dict.get(pid, self.zeros)
            features.append(feat)

        # Stack into [K, 1280] tensor
        return torch.stack(features), label
