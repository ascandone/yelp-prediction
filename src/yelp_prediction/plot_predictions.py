import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import dataframes
from MilModel import MILModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features/features.pt")
    parser.add_argument("--model", type=str, default="best_mil_model.pth")
    parser.add_argument("--outdir", type=str, default="reports/figures")
    parser.add_argument("--max_photos", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load data (same query as training)
    # -----------------------------
    df = dataframes.q_features.collect()
    np.random.seed(args.seed)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df.filter(mask)
    val_df = df.filter(~mask)

    # Median from FULL dataset (matches current train_features.py behavior)
    # If you later fix leakage, compute this from train_df instead.
    median = df["stars"].median()

    # -----------------------------
    # Load features + model
    # -----------------------------
    features_path = Path(args.features)
    model_path = Path(args.model)

    if not features_path.exists():
        raise FileNotFoundError(
            f"No encuentro features en {features_path}. "
            f"¿Corriste compute_features.py y guardaste data/features/features.pt?"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No encuentro el modelo en {model_path}. "
            f"¿Entrenaste y generaste best_mil_model.pth?"
        )

    features_dict = torch.load(features_path, map_location="cpu")

    model = MILModel(median_stars=median).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # -----------------------------
    # Predict on val_df deterministically
    # We'll aggregate using the SAME logic as MILModel: mean over K photos.
    # Here we take up to max_photos photos per business (deterministic with seed).
    # -----------------------------
    val_rows = val_df.to_dicts()
    business_ids = [r["business_id"] for r in val_rows]
    y_true = np.array([float(r["stars"]) for r in val_rows], dtype=np.float32)

    # deterministic sampling
    rng = np.random.default_rng(args.seed)

    input_dim = 1280
    X = torch.zeros((len(val_rows), args.max_photos, input_dim), dtype=torch.float32)

    zeros = torch.zeros(input_dim, dtype=torch.float32)
    n_available = np.zeros(len(val_rows), dtype=np.int32) 
    n_used = np.zeros(len(val_rows), dtype=np.int32)

    for i, r in enumerate(val_rows):
        photo_ids = r.get("photo_ids") or []

        # Número de fotos disponibles (del bag completo)
        n_avail = int(r.get("photo_count", len(photo_ids)))
        n_available[i] = n_avail

        if n_avail == 0:
            continue

        if n_avail >= args.max_photos:
            selected = rng.choice(photo_ids, size=args.max_photos, replace=False)
            n_used[i] = args.max_photos
        else:
            selected = rng.choice(photo_ids, size=args.max_photos, replace=True)
            n_used[i] = n_avail  # únicas disponibles

        feats = [features_dict.get(pid, zeros) for pid in selected]
        X[i] = torch.stack(feats)


    # batch inference
    y_pred_list = []
    with torch.inference_mode():
        for start in range(0, len(X), args.batch_size):
            xb = X[start : start + args.batch_size].to(DEVICE)
            preds = model(xb).squeeze(-1).detach().cpu().numpy()
            y_pred_list.append(preds)

    y_pred = np.concatenate(y_pred_list).astype(np.float32)

    # -----------------------------
    # Metrics
    # -----------------------------
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Save predictions CSV
    pred_csv = outdir / "val_predictions.csv"
    with pred_csv.open("w", encoding="utf-8") as f:
        f.write("business_id,y_true,y_pred,error,abs_error\n")
        for bid, yt, yp in zip(business_ids, y_true, y_pred):
            e = float(yp - yt)
            f.write(f"{bid},{yt:.4f},{yp:.4f},{e:.4f},{abs(e):.4f}\n")

    print(f"Validation size: {len(y_true)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved: {pred_csv}")

    # -----------------------------
    # Plots
    # -----------------------------

    # 1) Scatter: y_true vs y_pred
    import matplotlib.colors as colors
    plt.figure()

    # LogNorm ayuda porque photo_count suele ser heavy-tailed
    vmax = int(max(1, n_available.max()))
    if vmax <= 1:
        norm = None
    else:
        norm = colors.LogNorm(vmin=1, vmax=vmax)

    norm = colors.LogNorm(vmin=1, vmax=vmax)

    sc = plt.scatter(y_true, y_pred, c=np.clip(n_available, 1, None), s=8, alpha=0.5, norm=norm)
    plt.plot([1, 5], [1, 5])
    plt.xlabel("True stars")
    plt.ylabel("Predicted stars")
    plt.title(f"True vs Predicted (val) colored by #photos | MAE={mae:.3f}, RMSE={rmse:.3f}")

    cb = plt.colorbar(sc)
    cb.set_label("Available photos per business (log scale)")

    plt.tight_layout()
    plt.savefig(outdir / "scatter_true_vs_pred_colored_photos.png", dpi=200)
    plt.close()

    # 2) Histogram of errors
    plt.figure()
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction error (y_pred - y_true)")
    plt.ylabel("Count")
    plt.title("Error distribution (val)")
    plt.tight_layout()
    plt.savefig(outdir / "hist_error.png", dpi=200)
    plt.close()

    # 3) Calibration-style plot: average pred by true-star bins (rounded to 0.5)
    bins = np.round(y_true * 2) / 2.0
    uniq = np.unique(bins)
    means = []
    for u in uniq:
        means.append(float(np.mean(y_pred[bins == u])))

    plt.figure()
    plt.plot(uniq, means, marker="o")
    plt.plot([1, 5], [1, 5])
    plt.xlabel("True stars (binned to 0.5)")
    plt.ylabel("Avg predicted stars")
    plt.title("Binned calibration (val)")
    plt.tight_layout()
    plt.savefig(outdir / "calibration_binned.png", dpi=200)
    plt.close()

    mask_low = y_true < 3.0
    if mask_low.any():
        plt.figure()
        vmax = int(max(1, n_available[mask_low].max()))
        norm = colors.LogNorm(vmin=1, vmax=vmax)

        sc = plt.scatter(
            y_true[mask_low],
            y_pred[mask_low],
            c=np.clip(n_available[mask_low], 1, None),
            s=14,
            alpha=0.7,
            norm=norm,
        )
        plt.plot([1, 3], [1, 3])
        plt.xlabel("True stars (<3)")
        plt.ylabel("Predicted stars")
        plt.title("Low-rating focus (val) colored by #photos")

        cb = plt.colorbar(sc)
        cb.set_label("Available photos (log scale)")

        plt.tight_layout()
        plt.savefig(outdir / "scatter_lowstars_colored_photos.png", dpi=200)
        plt.close()

    print(f"Saved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
