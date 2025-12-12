# WARNING: vibe coded

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path


MAX_PHOTOS = 3  # Number of photos to sample per business


# TODO where should this belong to?
# Standard ImageNet stats
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class YelpBagDataset(Dataset):
    def __init__(
        self,
        dataframe,
        photo_dir: Path,
        transform=DEFAULT_TRANSFORM,
    ):
        self.photo_dir = photo_dir
        self.data = dataframe.to_dicts()
        self.transform = transform

        self.placeholder = torch.zeros(3, 224, 224)

    def _read_photo(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        photo_ids = row["photo_ids"]
        label = torch.tensor(row["stars"], dtype=torch.float32)

        # 1. SAMPLING: Pick K photos (Randomly sample if > K, repeat if < K)
        if len(photo_ids) >= MAX_PHOTOS:
            selected = np.random.choice(photo_ids, MAX_PHOTOS, replace=False)
        else:
            selected = np.random.choice(photo_ids, MAX_PHOTOS, replace=True)

        # 2. LOAD IMAGES
        images = []
        for pid in selected:
            try:
                with Image.open(self.photo_dir / f"{pid}.jpg") as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
            except Exception:
                # TODO use a neutral transformer
                # TODO emit warnings
                # Fallback if image is missing/corrupt
                # a black placeholder image
                placeholder = torch.zeros(3, 224, 224)
                images.append(placeholder)

        # Stack into [3, 3, 224, 224] tensor
        return torch.stack(images), label
