import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path


DEFAULT_MAX_PHOTOS = 3  # Number of photos to sample per business


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
        max_photos=DEFAULT_MAX_PHOTOS,
    ):
        self.photo_dir = photo_dir
        self.data = dataframe.to_dicts()
        self.transform = transform
        self.max_photos = max_photos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        photo_ids = row["photo_ids"]
        label = torch.tensor(row["stars"], dtype=torch.float32)

        # 1. SAMPLING: Pick K photos (Randomly sample if > K, repeat if < K)
        if len(photo_ids) >= self.max_photos:
            selected = np.random.choice(photo_ids, self.max_photos, replace=False)
        else:
            selected = np.random.choice(photo_ids, self.max_photos, replace=True)

        # Stack into [3, 3, 224, 224] tensor
        return torch.stack([self._load_photo(pid) for pid in selected]), label

    def _load_photo(self, pid):
        try:
            with Image.open(self.photo_dir / f"{pid}.jpg") as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img
        except Exception:
            # TODO use a neutral tensor
            # TODO emit warning
            # Fallback with a black placeholder image if image is missing/corrupt
            return torch.zeros(3, 224, 224)
