import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import polars as pl
from dataframes import PHOTOS_DIR

# --- CONFIG ---
BATCH_SIZE = 64  # Bigger batch size for inference since no gradients
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- DATASET ---
class RawPhotoDataset(Dataset):
    """
    Simple dataset to just iterate over all photos for extraction.
    Does NOT transform into bags. 1 item = 1 photo.
    """

    def __init__(self, photo_df: pl.DataFrame, photo_dir: Path):
        self.photo_ids = photo_df["photo_id"].to_list()
        self.photo_dir = photo_dir

        # Standard ImageNet transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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


def collate_fn(batch):
    """Stack batch"""
    imgs, pids = zip(*batch)
    return torch.stack(imgs), pids


def run():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading photo list...")
    df_photos = pl.scan_ndjson(PHOTOS_DIR / "photos.json").collect()
    print(f"Found {len(df_photos)} photos to process.")

    # 2. Setup Model (Backbone Only)
    print("Loading EfficientNetV2-S...")
    full_model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT
    )
    backbone = full_model.features
    backbone = backbone.to(DEVICE)
    backbone.eval()  # Eval mode is critical for BatchNorm/Dropout

    # Global pooling layer to flatten [B, 1280, 7, 7] -> [B, 1280]
    pool = torch.nn.AdaptiveAvgPool2d(1)

    # 3. Setup Loader
    dataset = RawPhotoDataset(df_photos, PHOTOS_DIR / "photos")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # 4. Inference Loop
    print("Starting extraction...")
    features_dict = {}

    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)

            # Forward Pass
            # Shape: [B, 1280, 7, 7]
            features = backbone(imgs)

            # Pool: [B, 1280, 1, 1] -> [B, 1280]
            features = pool(features).flatten(1)

            # Store in memory
            for i, pid in enumerate(pids):
                # Save as CPU tensor to save GPU memory
                features_dict[pid] = features[i].cpu().clone()

    print(f"\nDone! Extracted {len(features_dict)} features in memory.")
    return features_dict
