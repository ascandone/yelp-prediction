from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import dataframes
from dataset import SinglePhotoDataset
from model import SinglePhotoModel
import polars as pl
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(seed=42)

g = torch.Generator()
g.manual_seed(42)


class WeightedL1Loss(nn.Module):
    def __init__(self, class_weights: dict):
        super().__init__()
        self.weights = class_weights

    def forward(self, input, target):
        loss = torch.abs(input - target)

        batch_weights = []
        target_cpu = target.detach().cpu().numpy()
        for t in target_cpu:
            batch_weights.append(self.weights.get(t, 1.0))

        w_tensor = torch.tensor(batch_weights, device=input.device).float()
        return (loss * w_tensor).mean()


def run(
    features_dict: dict,
    *,
    epochs=20,
    batch_size=128,
    lr=0.001,
    max_photos=3,
    criterion=None,
    input_dim=512,
):
    """
    Train a MIL model on pre-computed features using Regression + Weighted L1 Loss.
    Metrics are aggregated by Business ID.
    """

    # Stats
    df = dataframes.q_features.collect()
    median = df["stars"].median()
    baseline_mae = (df["stars"] - median).abs().mean()

    print(f"Dataset Size: {len(df)} businesses")
    print(f"Median Stars: {median:.2f} (Global MAE baseline: {baseline_mae:.2f})")

    # Split
    mask = rng.random(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    # Datasets
    train_dataset = SinglePhotoDataset(train_df, features_dict)
    val_dataset = SinglePhotoDataset(val_df, features_dict)

    # Calculate Weights
    print("Computing weights...")
    train_targets = [item["stars"] for item in train_dataset.data]
    class_counts = Counter(train_targets)

    # 1. Sampler Weights
    sampler_weights_map = {k: 1.0 / np.sqrt(v) for k, v in class_counts.items()}
    sample_weights_vector = [sampler_weights_map[t] for t in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights_vector,
        num_samples=len(sample_weights_vector),
        replacement=True,
    )

    # 2. Loss Weights
    median_count = (
        class_counts[median]
        if median in class_counts
        else list(class_counts.values())[0]
    )
    loss_weights_map = {k: np.sqrt(median_count / v) for k, v in class_counts.items()}

    criterion = WeightedL1Loss(loss_weights_map)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=g,
    )

    # Model Setup
    model = SinglePhotoModel(median_stars=median, input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train Loop
    best_biz_macro_mae = float("inf")
    validation_output = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Training (Per Photo)
        # Note: We unpack 4 items now, but we ignore photo_id and business_id during training
        for feats, targets, _, _ in tqdm(
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

        # Validation (Aggregated by Business)
        model.eval()

        # Helper to aggregate: business_id -> [predictions]
        biz_preds_map = defaultdict(list)
        biz_target_map = {}  # business_id -> true_stars

        val_predictions_raw = []  # For saving to partial output if needed
        val_photo_ids_raw = []

        with torch.no_grad():
            for feats, targets, photo_ids, business_ids in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                preds = model(feats).squeeze()

                batch_preds = preds.cpu().numpy()
                batch_targets = targets.cpu().numpy()

                # Accumulate for aggregation
                for pid, bid, pred, targ in zip(
                    photo_ids, business_ids, batch_preds, batch_targets
                ):
                    biz_preds_map[bid].append(pred)
                    biz_target_map[bid] = (
                        targ  # Overwrite is fine, same biz has same target
                    )

                    val_predictions_raw.append(pred)
                    val_photo_ids_raw.append(pid)

        # Compute Business-Level Metrics
        final_biz_preds = []
        final_biz_targets = []

        for bid, preds_list in biz_preds_map.items():
            mean_pred = np.mean(preds_list)
            # Clip predictions to valid range [1, 5] just in case
            mean_pred = np.clip(mean_pred, 1.0, 5.0)
            final_biz_preds.append(mean_pred)
            final_biz_targets.append(biz_target_map[bid])

        final_biz_preds = np.array(final_biz_preds)
        final_biz_targets = np.array(final_biz_targets)

        # Global Metrics (Business Level)
        errors = np.abs(final_biz_preds - final_biz_targets)
        biz_mae = np.mean(errors)

        # Per-Class Metrics (Business Level)
        unique_stars = np.unique(final_biz_targets)
        class_maes = {}
        for s in unique_stars:
            mask_s = final_biz_targets == s
            if np.sum(mask_s) > 0:
                class_mae = np.mean(errors[mask_s])
                class_maes[s] = class_mae

        biz_macro_mae = np.mean(list(class_maes.values()))

        # Save raw output for posterity
        for pid, pred in zip(val_photo_ids_raw, val_predictions_raw):
            validation_output.append((epoch + 1, pid, pred))

        # Formatting print
        one_star_mae = class_maes.get(1.0, float("nan"))
        two_star_mae = class_maes.get(2.0, float("nan"))
        four_star_mae = class_maes.get(4.0, float("nan"))

        is_best = biz_macro_mae < best_biz_macro_mae
        best_label = " | Best Biz-Macro!" if is_best and epoch > 0 else ""

        print(
            f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
            f"Biz MAE: {biz_mae:.3f} | Biz Macro: {biz_macro_mae:.3f} | "
            f"1*: {one_star_mae:.3f} | 2*: {two_star_mae:.3f} | 4*: {four_star_mae:.3f}{best_label}"
        )

        if is_best:
            best_biz_macro_mae = biz_macro_mae
            torch.save(model.state_dict(), "data/best_model.pth")

    print(
        f"\nTraining Complete. Best Validation Business-Level Macro-MAE: {best_biz_macro_mae:.4f}"
    )

    path = Path("data/predictions.csv")
    print(f"Writing to '{path}'..")
    df = pl.DataFrame(
        validation_output,
        schema=["epoch", "photo_id", "prediction"],
        orient="row",
    )
    df.write_csv(path)
    print("Done. ✅")
