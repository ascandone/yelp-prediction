from typing import Literal
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import models
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
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


def _setup_efficientnet_v2_s():
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
    features_dict = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            features = backbone(imgs)
            features = pool(features).flatten(1)
            for i, pid in enumerate(pids):
                features_dict[pid] = features[i].cpu().clone()

    return features_dict


def _setup_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    def clip_transform(img):
        # Processor returns a dict with 'pixel_values' of shape [1, 3, 224, 224]
        # We squeeze to get [3, 224, 224]
        inputs = processor(images=img, return_tensors="pt")  # type: ignore
        return inputs["pixel_values"].squeeze(0)

    # Setup Loader
    dataset = RawPhotoDataset(
        photo_df=dataframes.q_photos.collect(),
        photo_dir=dataframes.PHOTOS_DIR / "photos",
        transform=clip_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=_collate_fn,
    )

    # Inference Loop
    features_dict = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            features = model.get_image_features(pixel_values=imgs)
            for i, pid in enumerate(pids):
                features_dict[pid] = features[i].cpu().clone()

    return features_dict


ModelType = Literal["clip"] | Literal["efficientnet-v2-s"]


def _get_model(model: ModelType) -> dict:
    match model:
        case "clip":
            return _setup_clip()
        case "efficientnet-v2-s":
            return _setup_efficientnet_v2_s()


def run(
    *,
    save_to_disk: bool = True,
    model: ModelType = "clip",
):
    """
    Extract features from photos and return them as a dictionary.
    By default (save_to_disk option), save the dictionary to disk.
    """

    print(f"Using device: {DEVICE}")

    print(f"Setting up model: {model}")
    features_dict = _get_model(model)

    print(f"\nDone! Extracted {len(features_dict)} features in memory.")

    if save_to_disk:
        out_path = OUTPUT_DIR / "features.pt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving features to {out_path}...")
        torch.save(features_dict, out_path)
        print("Saved.")

    return features_dict
