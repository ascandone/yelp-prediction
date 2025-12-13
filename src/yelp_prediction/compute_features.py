import torch
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import dataframes
from dataset import RawPhotoDataset

# --- CONFIG ---
BATCH_SIZE = 64  # Bigger batch size for inference since no gradients
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("data/features")


def _collate_fn(batch):
    """Stack batch"""
    imgs, pids = zip(*batch)
    return torch.stack(imgs), pids


def _setup_resnet():
    # Setup Model (Backbone Only)
    full_model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.DEFAULT,
    )
    backbone = full_model.features.to(DEVICE)
    backbone.eval()

    # Global pooling layer to flatten [B, 1280, 7, 7] -> [B, 1280]
    pool = torch.nn.AdaptiveAvgPool2d(1)

    # Setup Loader
    dataset = RawPhotoDataset(
        photo_df=dataframes.q_photos.collect(),
        photo_dir=dataframes.PHOTOS_DIR / "photos",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn,
    )

    # Inference Loop
    print("Starting extraction...")
    features_dict = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            features = backbone(imgs)
            features = pool(features).flatten(1)
            for i, pid in enumerate(pids):
                # Save as CPU tensor to save GPU memory
                features_dict[pid] = features[i].cpu().clone()

    return features_dict


def run(*, save_to_disk: bool = True):
    """
    Extract features from photos and return them as a dictionary.
    By default (save_to_disk option), save the dictionary to disk.
    """

    print(f"Using device: {DEVICE}")

    features_dict = _setup_resnet()

    print(f"\nDone! Extracted {len(features_dict)} features in memory.")

    if save_to_disk:
        out_path = OUTPUT_DIR / "features.pt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving features to {out_path}...")
        torch.save(features_dict, out_path)
        print("Saved.")

    return features_dict
