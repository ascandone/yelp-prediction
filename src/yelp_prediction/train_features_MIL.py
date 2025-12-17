from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import polars as pl
from model import MILModel  # Assumendo che sia già aggiornato con init_rating e attention pooling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset wrapper
# ---------------------------
class SinglePhotoMILDataset(Dataset):
    def __init__(self, df: pl.DataFrame, features_dict: dict, max_photos=3, input_dim=512):
        self.business_ids = df["business_id"].to_list()
        self.stars = df["stars"].to_list()
        self.features_dict = features_dict
        self.max_photos = max_photos
        self.input_dim = input_dim

    def __len__(self):
        return len(self.business_ids)

    def __getitem__(self, idx):
        bid = self.business_ids[idx]
        target = self.stars[idx]

        feats_list = self.features_dict.get(bid, [])
        feats_list = feats_list[:self.max_photos]

        # padding se necessario
        feats_list = [
            f if isinstance(f, list) else [float(x) for x in f]
            for f in feats_list
        ]
        feats_list += [[0.0]*self.input_dim] * (self.max_photos - len(feats_list))
        feats_tensor = torch.tensor(feats_list, dtype=torch.float32)  # (max_photos, input_dim)

        return feats_tensor, torch.tensor(target, dtype=torch.float32), bid

# ---------------------------
# Training MIL
# ---------------------------
def run(features_dict: dict, yelp_df: pl.DataFrame, *, epochs=20, batch_size=128, lr=0.001, max_photos=3, input_dim=512, attention_dim=256):
    # ---------------------------
    # Split train/val
    # ---------------------------
    np.random.seed(42)
    mask = np.random.rand(len(yelp_df)) < 0.8
    train_df = yelp_df.filter(mask)
    val_df = yelp_df.filter(~mask)

    train_avg = train_df["stars"].mean()
    print(f"Train: {len(train_df)} businesses, Avg stars: {train_avg:.4f}")
    print(f"Validation: {len(val_df)} businesses")

    # ---------------------------
    # DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        SinglePhotoMILDataset(train_df, features_dict, max_photos=max_photos, input_dim=input_dim),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        SinglePhotoMILDataset(val_df, features_dict, max_photos=max_photos, input_dim=input_dim),
        batch_size=batch_size,
        shuffle=False
    )

    # ---------------------------
    # Model & optimizer
    # ---------------------------
    model = MILModel(init_rating=float(train_avg), input_dim=input_dim, attention_dim=attention_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    best_mae = float("inf")
    validation_output = []

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for feats, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(feats).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        errors = []
        with torch.no_grad():
            for feats, targets, photo_ids in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                preds = model(feats).squeeze()

                for pid, pred in zip(photo_ids, preds.cpu().numpy()):
                    validation_output.append((epoch+1, pid, pred))

                errors.extend(torch.abs(preds - targets).cpu().numpy())

        val_mae = np.mean(errors)
        val_rmse = np.sqrt(np.mean(np.square(errors)))
        best_label = ""
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "data/best_model.pth")
            best_label = " | Best MAE!"

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}{best_label}")

    print(f"\nTraining complete. Best Validation MAE: {best_mae:.4f}")

    # ---------------------------
    # Save predictions
    # ---------------------------
    out_path = Path("data/predictions.csv")
    pl.DataFrame(validation_output, schema=["epoch", "photo_id", "prediction"], orient="row").write_csv(out_path)
    print(f"Predictions saved to {out_path}")
