import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import dataframes
from YelpFeatureDataset import YelpFeatureDataset
from MilModel import MILModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    features_dict: dict,
    *,
    epochs=20,
    batch_size=128,
    lr=0.001,
    max_photos=3,
    criterion=nn.MSELoss(),
    input_dim=1280,
):
    """
    Train a MIL model on pre-computed features (see compute_features.py)
    """

    # Stats
    df = dataframes.q_features.collect()
    median = df["stars"].median()
    baseline_mae = (df["stars"] - median).abs().mean()
    stdev = df["stars"].std()

    avg = df["stars"].mean()
    print(f"Dataset Size: {len(df)} businesses")
    print(f"Median Stars: {median:.2f} (baseline mae: {baseline_mae:.2f})")
    print(f"Avg Stars: {avg:.2f} (stdev: {stdev:.2f})")

    # Split
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    # Loaders
    train_loader = DataLoader(
        YelpFeatureDataset(
            train_df,
            features_dict,
            max_photos=max_photos,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        YelpFeatureDataset(
            val_df,
            features_dict,
            max_photos=max_photos,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Model Setup
    model = MILModel(median_stars=median, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train Loop
    best_mae = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for feats, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(feats).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        errors = []
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                preds = model(feats).squeeze()
                errors.extend(torch.abs(preds - targets).cpu().numpy())

        val_mae = np.mean(errors)
        val_rmse = np.sqrt(np.mean(np.square(errors)))
        is_best_score = val_mae < best_mae
        best_score_label = " | Best score" if is_best_score and epoch > 0 else ""
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}{best_score_label}"
        )

        if is_best_score:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_mil_model.pth")

    print(f"\nTraining Complete. Best Validation MAE: {best_mae:.4f}")
