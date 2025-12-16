from __future__ import annotations

from typing import Callable, Literal, Tuple
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

import dataframes
from dataset import RawPhotoDataset

# --- CONFIG ---
BATCH_SIZE = 64  # Bigger batch size for inference since no gradients
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("data/features")


def _collate_fn(batch):
    """Stack batch"""
    imgs, pids = zip(*batch)
    return torch.stack(imgs), pids


def _build_loader(*, transform: Callable | None):
    dataset = RawPhotoDataset(
        photo_df=dataframes.q_photos.collect(),
        photo_dir=dataframes.PHOTOS_DIR / "photos",
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate_fn,
    )


def _setup_efficientnet_v2_s() -> dict:
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    full_model = models.efficientnet_v2_s(weights=weights)
    backbone = full_model.features.to(DEVICE)
    backbone.eval()

    # Global pooling layer to flatten [B, 1280, 7, 7] -> [B, 1280]
    pool = nn.AdaptiveAvgPool2d(1)

    loader = _build_loader(transform=weights.transforms())

    # Inference Loop
    features_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features (efficientnet-v2-s)"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            features = backbone(imgs)
            features = pool(features).flatten(1).detach().cpu()

            for pid, feat in zip(pids, features):
                features_dict[pid] = feat.clone()

    return features_dict


def _setup_clip() -> dict:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    def clip_transform(img):
        # Processor returns a dict with 'pixel_values' of shape [1, 3, 224, 224]
        inputs = processor(images=img, return_tensors="pt")  # type: ignore
        return inputs["pixel_values"].squeeze(0)

    loader = _build_loader(transform=clip_transform)

    # Inference Loop
    features_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc="Extracting Features (clip)"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            features = model.get_image_features(pixel_values=imgs).detach().cpu()

            for pid, feat in zip(pids, features):
                features_dict[pid] = feat.clone()

    return features_dict


def make_torchvision_backbone(name: str) -> Tuple[torch.nn.Module, Callable, int]:
    """Return (backbone, transform, output_dim) for torchvision backbones."""
    name = name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=weights)
        m.fc = nn.Identity()
        transform = weights.transforms()
        dim = 512

    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        m = models.resnet50(weights=weights)
        m.fc = nn.Identity()
        transform = weights.transforms()
        dim = 2048

    elif name == "resnet152":
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        m = models.resnet152(weights=weights)
        m.fc = nn.Identity()
        transform = weights.transforms()
        dim = 2048

    elif name == "vgg19":
        weights = models.VGG19_Weights.IMAGENET1K_V1
        m = models.vgg19(weights=weights)
        # Replace final classifier layer: [B, 4096] instead of logits
        m.classifier[-1] = nn.Identity()
        transform = weights.transforms()
        dim = 4096

    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        m = models.vit_b_16(weights=weights)
        # Remove classification head: [B, 768] embedding
        m.heads = nn.Identity()
        transform = weights.transforms()
        dim = 768

    else:
        raise ValueError(f"Unknown torchvision backbone: {name}")

    m.eval()
    return m, transform, dim


def _setup_torchvision(model_name: str) -> dict:
    backbone, transform, dim = make_torchvision_backbone(model_name)
    backbone = backbone.to(DEVICE)
    backbone.eval()

    loader = _build_loader(transform=transform)

    features_dict: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for imgs, pids in tqdm(loader, desc=f"Extracting Features ({model_name})"):
            if len(imgs) == 0:
                continue

            imgs = imgs.to(DEVICE)
            feats = backbone(imgs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.detach().cpu()

            for pid, feat in zip(pids, feats):
                features_dict[pid] = feat.clone()

    # Dim is printed for convenience (training needs to set input_dim accordingly)
    print(f"[{model_name}] feature dim = {dim}")
    return features_dict


ModelType = Literal[
    "clip",
    "efficientnet-v2-s",
    "resnet18",
    "resnet50",
    "resnet152",
    "vgg19",
    "vit_b_16",
]


def _get_model(model: ModelType) -> dict:
    match model:
        case "clip":
            return _setup_clip()
        case "efficientnet-v2-s":
            return _setup_efficientnet_v2_s()
        case "resnet18" | "resnet50" | "resnet152" | "vgg19" | "vit_b_16":
            return _setup_torchvision(model)
        case _:
            raise ValueError(f"Unsupported model: {model}")


def _output_path(model: str) -> Path:
    # Consistent naming so main/experiments can load features by backbone.
    return OUTPUT_DIR / f"features-{model}.pt"


def run(*, save_to_disk: bool = True, model: ModelType = "clip") -> dict:
    """Extract features from photos and return them as a dictionary.

    By default (save_to_disk=True), saves the dictionary to disk at:
    data/features/features-<model>.pt
    """

    print(f"Using device: {DEVICE}")
    print(f"Setting up model: {model}")

    features_dict = _get_model(model)

    print(f"\nDone! Extracted {len(features_dict)} features in memory.")

    if save_to_disk:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _output_path(model)
        print(f"Saving features to {out_path}...")
        torch.save(features_dict, out_path)
        print("Saved.")

    return features_dict


def _parse_args():
    p = argparse.ArgumentParser(description="Precompute image embeddings for Yelp photos")
    p.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=[
            "clip",
            "efficientnet-v2-s",
            "resnet18",
            "resnet50",
            "resnet152",
            "vgg19",
            "vit_b_16",
        ],
        help="Backbone to use for feature extraction",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save features to disk (return only)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(save_to_disk=(not args.no_save), model=args.model)  # type: ignore[arg-type]
