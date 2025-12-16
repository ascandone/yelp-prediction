from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataframes
from dataset import SinglePhotoDataset
from model import SinglePhotoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    features_dict: dict,
    *,
    model_tag: str = "clip",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    max_photos: int = 3,  # kept for compatibility; SinglePhotoDataset ignores it
    criterion=None,
    input_dim: int = 512,  # clip default
    split_seed: int = 42,
    save_dir: str = "data",
):
    '''
    Train a model on pre-computed photo features (see compute_features.py).

    Notes:
      - This script trains a SinglePhotoModel (one prediction per photo).
      - Outputs are written with `model_tag` in the filename so you can compare backbones.
    '''
    if criterion is None:
        criterion = nn.L1Loss()

    save_dir = Path(save_dir)
    (save_dir / "models").mkdir(parents=True, exist_ok=True)
    (save_dir / "preds").mkdir(parents=True, exist_ok=True)

    # Stats
    df = dataframes.q_features.collect()
    median = df["stars"].median()
    baseline_mae = (df["stars"] - median).abs().mean()
    stdev = df["stars"].std()
    avg = df["stars"].mean()

    print(f"\n[train_features] model_tag={model_tag} | input_dim={input_dim} | device={DEVICE}")
    print(f"Dataset Size: {len(df)} businesses")
    print(f"Median Stars: {median:.2f} (baseline mae: {baseline_mae:.2f})")
    print(f"Avg Stars: {avg:.2f} (stdev: {stdev:.2f})\n")

    # Split (reproducible)
    rng = np.random.default_rng(split_seed)
    mask = rng.random(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    # Loaders
    train_loader = DataLoader(
        SinglePhotoDataset(train_df, features_dict),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SinglePhotoDataset(val_df, features_dict),
        batch_size=batch_size,
        shuffle=False,
    )

    # Model
    # (kept as-is from your original script: you pass avg into the parameter named median_stars)
    model = SinglePhotoModel(median_stars=avg, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae = float("inf")
    best_rmse = float("inf")
    best_epoch = -1
    best_val_output = []  # (photo_id, prediction)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0

        for feats, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(feats).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        abs_errors = []
        sq_errors = []
        epoch_output = []

        with torch.no_grad():
            for feats, targets, photo_ids in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                preds = model(feats).squeeze()

                diff = preds - targets
                abs_errors.extend(torch.abs(diff).cpu().numpy())
                sq_errors.extend((diff ** 2).cpu().numpy())

                batch_preds = preds.detach().cpu().numpy()
                for pid, pred in zip(photo_ids, batch_preds):
                    epoch_output.append((pid, float(pred)))

        val_mae = float(np.mean(abs_errors)) if abs_errors else float("nan")
        val_rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")

        is_best = val_mae < best_mae
        best_label = " | Best" if is_best and epoch > 0 else ""
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}{best_label}"
        )

        if is_best:
            best_mae = val_mae
            best_rmse = val_rmse
            best_epoch = epoch + 1
            best_val_output = epoch_output

            model_path = save_dir / "models" / f"best_model_{model_tag}.pth"
            torch.save(model.state_dict(), model_path)

    print(f"\nTraining Complete. Best Val MAE: {best_mae:.4f} | RMSE: {best_rmse:.4f} | epoch={best_epoch}")

    # Write predictions for BEST epoch only (smaller and easier to compare)
    pred_path = save_dir / "preds" / f"predictions_{model_tag}.csv"
    print(f"Writing best-epoch predictions to '{pred_path}'..")
    out_df = pl.DataFrame(best_val_output, schema=["photo_id", "prediction"], orient="row")
    out_df.write_csv(pred_path)
    print("Done. ✅")

    return {
        "model_tag": model_tag,
        "input_dim": int(input_dim),
        "best_epoch": int(best_epoch),
        "best_mae": float(best_mae),
        "best_rmse": float(best_rmse),
        "model_path": str(save_dir / "models" / f"best_model_{model_tag}.pth"),
        "pred_path": str(pred_path),
        "split_seed": int(split_seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
    }
