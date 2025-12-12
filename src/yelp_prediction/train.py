import numpy as np
import torch
from RatingPredictor import RatingPredictor
from dataframes import df_final, PHOTOS_DIR
from torch.utils.data import DataLoader
from YelpBagDataset import YelpBagDataset
import torch.nn as nn
from tqdm import tqdm

BATCH_SIZE = 32  # T4 handles 32 easily with frozen backbone
LR = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    df = df_final

    median_stars = df["stars"].median()
    baseline_mae = (df["stars"] - median_stars).abs().mean()

    print(f"Baseline to beat: {baseline_mae:.2f} (median: {median_stars})")

    model = RatingPredictor(
        median_stars=median_stars,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Split 80/20

    mask = np.random.rand(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    photos_dir = PHOTOS_DIR / "photos"

    train_loader = DataLoader(
        YelpBagDataset(
            train_df,
            photo_dir=photos_dir,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        YelpBagDataset(
            val_df,
            photo_dir=photos_dir,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    # WARNING: vibe coded

    print("Starting Training...")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0

        # Progress bar for sanity check
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

        # --- QUICK EVAL ---
        model.eval()
        errors = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                errors.extend(torch.abs(preds - targets).cpu().numpy())

        print(f"--> Validation MAE: {np.mean(errors):.4f} stars\n")

    print("done!")
