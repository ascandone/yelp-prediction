import torch
from torch.utils.data import Dataset
import numpy as np


class YelpFeatureDataset(Dataset):
    FEATURE_DIM = 1280  # EfficientNetV2-S output size
    DEFAULT_MAX_PHOTOS = 3

    def __init__(
        self,
        dataframe,
        features_dict: dict,
        max_photos=YelpFeatureDataset.DEFAULT_MAX_PHOTOS,
    ):
        self.features_dict = features_dict
        self.max_photos = max_photos
        # Convert to dictionary for fast access
        self.data = dataframe.to_dicts()

        # Pre-allocate a zero vector for missing data
        self.zeros = torch.zeros(YelpFeatureDataset.FEATURE_DIM)

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
            return torch.zeros(self.max_photos, YelpFeatureDataset.FEATURE_DIM), label

        if len(photo_ids) >= self.max_photos:
            selected = np.random.choice(photo_ids, self.max_photos, replace=False)
        else:
            selected = np.random.choice(photo_ids, self.max_photos, replace=True)

        # 2. LOAD TENSORS
        features = []
        for pid in selected:
            # Fast in-memory lookup
            feat = self.features_dict.get(pid, self.zeros)
            features.append(feat)

        # Stack into [K, 1280] tensor
        return torch.stack(features), label
