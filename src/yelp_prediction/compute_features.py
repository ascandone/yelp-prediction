import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
import polars as pl
import dataframes
from dataset import RawPhotoDataset

# --- CONFIG ---
BATCH_SIZE = 64  # Bigger batch size for inference since no gradients
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("data/features")


def collate_fn(batch):
    """Stack batch"""
    imgs, pids = zip(*batch)
    return torch.stack(imgs), pids


def run(*, save_to_disk: bool = True):
    """
    Extract features from photos and return them as a dictionary.
    By default (save_to_disk option), save the dictionary to disk.
    """

    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading photo list...")
    df_photos = pl.scan_ndjson(dataframes.PHOTOS_DIR / "photos.json").collect()
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
    dataset = RawPhotoDataset(
        df_photos,
        dataframes.PHOTOS_DIR / "photos",
    )
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

    if save_to_disk:
        out_path = OUTPUT_DIR / "features.pt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving features to {out_path}...")
        torch.save(features_dict, out_path)
        print("Saved.")

    return features_dict
