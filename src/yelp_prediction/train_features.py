import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataframes
from YelpFeatureDataset import YelpFeatureDataset


class MILModel(nn.Module):
    def __init__(self, median_stars=4.0):
        super().__init__()

        # Simple Regression Head
        self.head = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 1)
        )

        # Exact match to RatingPredictor.py Logic
        # careful: this is technically incorrect: we're using the median of the
        # whole dataset, not just our split
        initial_bias = torch.logit(torch.tensor((median_stars - 1) / 4))
        nn.init.constant_(self.head[-1].bias, initial_bias)

    def forward(self, x):
        # x shape: [Batch, K_Photos, 1280]

        # 1. MEAN POOLING (The "Bag" Aggregation)
        # Average across the K photos dimension
        bag_feature = torch.mean(x, dim=1)  # -> [Batch, 1280]

        # 2. Regression
        raw_output = self.head(bag_feature)

        # 3. Exact match to RatingPredictor.py Sigmoid
        # Sigmoid -> [0, 1] * 4 -> [0, 4] + 1 -> [1, 5]
        return torch.sigmoid(raw_output) * 4 + 1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    features_dict: dict,
    *,
    epochs=20,
    batch_size=128,
    lr=0.001,
):
    """
    Train a MIL model on pre-computed features (see compute_features.py)
    """

    print("--- Fast MIL Training on Pre-computed Features ---")

    # Prepare Data
    df = dataframes.q_features.collect()
    median = df["stars"].median()
    print(f"Dataset Size: {len(df)} businesses")
    print(f"Median Stars: {median}")

    # Split
    mask = np.random.rand(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    # Loaders
    train_loader = DataLoader(
        YelpFeatureDataset(train_df, features_dict),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        YelpFeatureDataset(val_df, features_dict),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Model Setup
    model = MILModel(median_stars=median).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_mil_model.pth")

    print(f"\nTraining Complete. Best Validation MAE: {best_mae:.4f}")
