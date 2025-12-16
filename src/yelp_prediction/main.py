import argparse
from pathlib import Path

import torch
import train_features

# Embedding dimensionality per backbone (must match how features were extracted)
INPUT_DIMS = {
    "clip": 512,
    "efficientnet-v2-s": 1280,
    "resnet18": 512,
    "resnet50": 2048,
    "resnet152": 2048,
    "vgg19": 4096,
    "vit_b_16": 768,
}


def _resolve_features_path(model: str) -> Path:
    '''Prefer: data/features/features-<model>.pt. Fall back to legacy features.pt.'''
    candidates = [
        Path("data/features") / f"features-{model}.pt",
        Path("data/features") / "features.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No features file found. Expected one of:\n"
        + "\n".join([f"  - {c}" for c in candidates])
        + "\n\nTip: run compute_features.py first to generate features."
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train rating model from precomputed photo features.")
    p.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=sorted(INPUT_DIMS.keys()),
        help="Backbone used to compute features (determines file name and input_dim).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--max-photos",
        type=int,
        default=5,
        help="Kept for compatibility (SinglePhotoDataset ignores it).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    features_path = _resolve_features_path(args.model)
    print(f"Loading features from: {features_path}")
    features = torch.load(features_path, map_location="cpu")

    metrics = train_features.run(
        features_dict=features,
        model_tag=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_photos=args.max_photos,
        input_dim=INPUT_DIMS[args.model],
        split_seed=args.split_seed,
    )

    print("\nRun summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
