import torch
from torch.utils.data import Dataset
import numpy as np
import polars as pl
from PIL import Image
from torchvision import transforms
from pathlib import Path


class YelpFeatureDataset(Dataset):
    DEAULT_FEATURE_DIM = 512  # CLIP features dim
    DEFAULT_MAX_PHOTOS = 3

    def __init__(
        self,
        dataframe,
        features_dict: dict,
        max_photos=DEFAULT_MAX_PHOTOS,
    ):
        self.features_dict = features_dict
        self.max_photos = max_photos
        self.data = dataframe.to_dicts()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        photo_ids = row["photo_ids"]
        label = torch.tensor(row["stars"], dtype=torch.float32)

        if len(photo_ids) == 0:
            # This should be forbidden by preconditions
            raise Exception("No photos!")

        # SAMPLING Strategy suitable for MIL
        if len(photo_ids) >= self.max_photos:
            selected = np.random.choice(photo_ids, self.max_photos, replace=False)
        else:
            selected = np.random.choice(photo_ids, self.max_photos, replace=True)

        features = [self.features_dict[pid] for pid in selected]
        return torch.stack(features), label, str(selected.tolist())


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class RawPhotoDataset(Dataset):
    """
    Simple dataset to just iterate over all photos for extraction.
    Does NOT transform into bags. 1 item = 1 photo.
    """

    def __init__(
        self,
        photo_df: pl.DataFrame,
        photo_dir: Path,
        transform=DEFAULT_TRANSFORM,
    ):
        self.photo_ids = photo_df["photo_id"].to_list()
        self.photo_dir = photo_dir
        self.transform = transform

    def __len__(self):
        return len(self.photo_ids)

    def __getitem__(self, idx):
        pid = self.photo_ids[idx]
        path = self.photo_dir / f"{pid}.jpg"

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = self.transform(img)
                return img, pid
        except Exception:
            # Fallback to zeros (Matches Baseline YelpBagDataset)
            return torch.zeros(3, 224, 224), pid


class SinglePhotoDataset(Dataset):
    """
    Dataset for single-photo prediction (non-MIL approach).

    Unlike YelpFeatureDataset which returns bags of photos per business,
    this dataset "unrolls" the data so that each item is a single photo
    and its associated business rating.

    This means if a business has 5 photos, it will contribute 5 training samples.
    """

    def __init__(
        self,
        dataframe,
        features_dict: dict,
    ):
        """
        Args:
            dataframe: Polars DataFrame with columns ['business_id', 'photo_ids', 'stars']
            features_dict: Dictionary mapping photo_id -> feature tensor
        """
        self.features_dict = features_dict

        # Unroll the data: create (photo_id, stars) pairs
        self.data = []
        for row in dataframe.to_dicts():
            photo_ids = row["photo_ids"]
            stars = row["stars"]
            business_id = row["business_id"]

            # Add each photo as a separate training sample
            for photo_id in photo_ids:
                self.data.append(
                    {
                        "photo_id": photo_id,
                        "stars": stars,
                        "business_id": business_id,
                    }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        photo_id = item["photo_id"]
        business_id = item["business_id"]
        label = torch.tensor(item["stars"], dtype=torch.float32)

        # Load feature vector for this photo
        feature = self.features_dict[photo_id]

        # Return single photo feature (not a bag)
        # Shape: [FEATURE_DIM] instead of [K, FEATURE_DIM]
        return feature, label, photo_id, business_id
