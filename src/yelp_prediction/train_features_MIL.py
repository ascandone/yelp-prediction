from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import polars as pl
from model import MILModel  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        if len(feats_list) > 0:
            current_feats = torch.stack(feats_list)
        else:
            current_feats = torch.zeros((0, self.input_dim))

       
        padding_size = self.max_photos - current_feats.size(0)
        if padding_size > 0:
            padding = torch.zeros((padding_size, self.input_dim))
            feats_tensor = torch.cat([current_feats, padding], dim=0)
        else:
            feats_tensor = current_feats

        return feats_tensor, torch.tensor(target, dtype=torch.float32), bid


def run_MIL(features_dict: dict, yelp_df: pl.DataFrame, *, epochs=20, batch_size=128, lr=0.0001, max_photos=3, input_dim=512, attention_dim=256):
    
    np.random.seed(42)
    mask = np.random.rand(len(yelp_df)) < 0.8
    train_df = yelp_df.filter(mask)
    val_df = yelp_df.filter(~mask)

    train_avg = train_df["stars"].mean()
    baseline_errors = (val_df["stars"] - train_avg).abs()
    baseline_mae = baseline_errors.mean()


    print("-" * 30)
    print(f" ANALISI BASELINE")
    print(f"Media stelle (Train): {train_avg:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f} (Senza usare le foto)")
    print(f"Miglior MAE del tuo Modello: 0.4619")
    print("-" * 30)

    improvement = ((baseline_mae - 0.4619) / baseline_mae) * 100
    print(f"Miglioramento rispetto alla statistica: {improvement:.2f}%")

    print(f"Train: {len(train_df)} businesses, Avg stars: {train_avg:.4f}")
    print(f"Validation: {len(val_df)} businesses")


    
    y_train = train_df["stars"].to_numpy()
    y_train_rounded = np.round(y_train).astype(int)

    class_counts = np.bincount(y_train_rounded)
    class_counts = np.where(class_counts == 0, 1, class_counts) 


    weights_per_class = 1. / np.sqrt(class_counts)  
    sample_weights = weights_per_class[y_train_rounded]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

 
    train_loader = DataLoader(
        SinglePhotoMILDataset(train_df, features_dict, max_photos=max_photos, input_dim=input_dim),
        batch_size=batch_size,
        sampler = sampler,
        shuffle=False
    )
    val_loader = DataLoader(
        SinglePhotoMILDataset(val_df, features_dict, max_photos=max_photos, input_dim=input_dim),
        batch_size=batch_size,
        shuffle=False
    )

   
    model = MILModel(init_rating=float(train_avg), input_dim=input_dim, attention_dim=attention_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta = 0.5)

    best_mae = float("inf")
    validation_output = []


    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for feats, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            
            if model.training:

                noise = torch.randn_like(feats) * 0.01 
                feats = feats + noise


            optimizer.zero_grad()
            preds = model(feats).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

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

    out_path = Path("data/predictions.csv")
    pl.DataFrame(validation_output, schema=["epoch", "photo_id", "prediction"], orient="row").write_csv(out_path)
    print(f"Predictions saved to {out_path}")
