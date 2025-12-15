from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataframes
from dataset import SinglePhotoDataset
from model import SinglePhotoModel
import polars as pl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(
    features_dict: dict,
    *,
    epochs=20,
    batch_size=128,
    lr=0.001,
    max_photos=3,
    criterion=nn.L1Loss(),
    input_dim=512,  # clip
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
    train_ds = SinglePhotoDataset(
        train_df,
        features_dict,
    )

    # --- WeightedRandomSampler Implementation ---
    # 1. Extract targets (stars) from the dataset
    targets = []
    for item in train_ds.data:
        targets.append(item["stars"])
    targets = np.array(targets)

    # 2. Bin targets to integers for class weighting
    # Rounding: 1.0-1.49 -> 1, 1.5-2.49 -> 2, etc.
    # But effectively just round() works fine for simple buckets.
    # Note: Yelp stars are often 1.0, 1.5, 2.0...
    # Let's treat unique values as classes or just bins.
    # Simple approach: Round to nearest int: 1, 2, 3, 4, 5
    dataset_classes = np.round(targets).astype(int)

    # 3. Calculate weight for each class
    class_counts = np.bincount(dataset_classes)
    # Avoid division by zero if a class is missing (though unlikely)
    # Set weight 0 for index 0 (not used)
    class_weights = np.zeros_like(class_counts, dtype=np.float32)

    # Compute inverse frequency
    # We only care about indices 1..5
    for c in range(1, len(class_counts)):
        if class_counts[c] > 0:
            class_weights[c] = 1.0 / class_counts[c]

    # 4. Assign weight to each sample
    sample_weights = class_weights[dataset_classes]

    from torch.utils.data import WeightedRandomSampler

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("Using WeightedRandomSampler to balance classes:")
    print(f"Class Counts: {dict(enumerate(class_counts))}")
    print(f"Class Weights: {dict(enumerate(np.round(class_weights, 4)))}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Mutually exclusive with sampler
        sampler=sampler,
    )

    val_loader = DataLoader(
        SinglePhotoDataset(
            val_df,
            features_dict,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    # Model Setup
    model = SinglePhotoModel(median_stars=avg, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train Loop
    best_mae = float("inf")

    validation_output = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for feats, targets, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}",
            leave=False,
        ):

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
            for feats, targets, photo_ids in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                preds = model(feats).squeeze()

                batch_preds = preds.cpu().numpy()
                for pid, pred in zip(photo_ids, batch_preds):
                    validation_output.append((epoch + 1, pid, pred))

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
            torch.save(model.state_dict(), "data/best_model.pth")

    print(f"\nTraining Complete. Best Validation MAE: {best_mae:.4f}")

    path = Path("data/predictions.csv")
    print(f"Writing to '{path}'..")
    df = pl.DataFrame(
        validation_output,
        schema=["epoch", "photo_id", "prediction"],
        orient="row",
    )
    df.write_csv(path)
    print("Done. ✅")
