from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Preferred: package import (works in notebooks if 'src' is on sys.path)
try:  # pragma: no cover
    from yelp_prediction import dataframes
    from yelp_prediction.MilModel import MILModel
except ModuleNotFoundError:  # pragma: no cover
    # Fallback: run as a script from repo root
    #   py src\yelp_prediction\plot_predictions.py
    from src.yelp_prediction import dataframes  # type: ignore
    from src.yelp_prediction.MilModel import MILModel  # type: ignore


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EvalConfig:
    features_path: Path
    model_path: Path
    outdir: Path
    max_photos: int = 3
    batch_size: int = 256
    seed: int = 42
    train_frac: float = 0.8


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_train_val_split(df, *, seed: int = 42, train_frac: float = 0.8):
    """
    Train/val split consistent with the original training module:
    we sample a boolean mask with a fixed NumPy RNG seed.
    """
    np.random.seed(seed)
    mask = np.random.rand(len(df)) < train_frac
    train_df = df.filter(mask)
    val_df = df.filter(~mask)
    return train_df, val_df


def load_features_df():
    """
    Loads the "feature rows" LazyFrame from `dataframes.q_features` and materializes it.
    """
    return dataframes.q_features.collect()


def compute_median_stars(df) -> float:
    """
    Median star value used for model centering.
    Note: this is computed on the full df (train + val) to preserve current behavior.
    If you want to avoid leakage, compute it on train_df only.
    """
    return float(df["stars"].median())


def load_features_dict(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path}. "
            f"Did you run compute_features.py to produce data/features/features.pt?"
        )
    return torch.load(path, map_location="cpu")


def load_model(model_path: Path, *, median_stars: float, device=DEVICE) -> MILModel:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Did you train and save best_mil_model.pth?"
        )
    model = MILModel(median_stars=median_stars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _select_photo_ids(
    photo_ids: list[str],
    *,
    max_photos: int,
    rng: np.random.Generator,
    sampling: Literal["random", "deterministic"] = "random",
) -> tuple[list[str], int]:
    """
    Select up to `max_photos` photo_ids from the available set.

    - random: match current behavior (rng.choice; replace if needed)
    - deterministic: take the first K in sorted order; repeat if needed
    """
    n_avail = len(photo_ids)
    if n_avail == 0:
        return [], 0

    if sampling == "deterministic":
        ordered = sorted(photo_ids)
        if n_avail >= max_photos:
            selected = ordered[:max_photos]
        else:
            # repeat deterministically to reach max_photos
            reps = (max_photos + n_avail - 1) // n_avail
            selected = (ordered * reps)[:max_photos]
        return selected, n_avail

    # sampling == "random" (default)
    if n_avail >= max_photos:
        selected = rng.choice(photo_ids, size=max_photos, replace=False).tolist()
    else:
        selected = rng.choice(photo_ids, size=max_photos, replace=True).tolist()
    return selected, n_avail


def predict_val(
    val_df,
    *,
    features_dict: dict,
    model: MILModel,
    max_photos: int = 3,
    batch_size: int = 256,
    seed: int = 42,
    sampling: Literal["random", "deterministic"] = "random",
    device=DEVICE,
) -> dict[str, np.ndarray]:
    """
    Run inference on val_df by generating a tensor [N, K, 1280], where K=max_photos.

    Returns arrays:
      - business_id
      - y_true, y_pred
      - error, abs_error
      - n_available (available photos)
      - n_used (photos used by the model)
      - photo_count_col (optional column if present in q_features)
    """
    rows = val_df.to_dicts()

    business_ids = np.array([r["business_id"] for r in rows], dtype=object)
    y_true = np.array([float(r["stars"]) for r in rows], dtype=np.float32)

    rng = np.random.default_rng(seed)
    input_dim = 1280
    zeros = torch.zeros(input_dim, dtype=torch.float32)

    n_available = np.zeros(len(rows), dtype=np.int32)
    n_used = np.zeros(len(rows), dtype=np.int32)

    # If q_features contains photo_count, retain it for diagnostics
    photo_count_col = np.array([int(r.get("photo_count", 0) or 0) for r in rows], dtype=np.int32)

    y_pred_out = np.zeros(len(rows), dtype=np.float32)

    def batch_iter():
        for start in range(0, len(rows), batch_size):
            end = min(len(rows), start + batch_size)
            xb = torch.zeros((end - start, max_photos, input_dim), dtype=torch.float32)

            for i, r in enumerate(rows[start:end]):
                photo_ids = r.get("photo_ids") or []
                selected, n_avail = _select_photo_ids(
                    photo_ids,
                    max_photos=max_photos,
                    rng=rng,
                    sampling=sampling,
                )

                n_available[start + i] = n_avail
                n_used[start + i] = min(n_avail, max_photos) if sampling == "random" else min(max_photos, max(1, n_avail))

                if n_avail == 0:
                    continue

                feats = [features_dict.get(pid, zeros) for pid in selected]
                xb[i] = torch.stack(feats)

            yield start, end, xb

    with torch.inference_mode():
        for start, end, xb in batch_iter():
            preds = model(xb.to(device)).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            y_pred_out[start:end] = preds

    error = y_pred_out - y_true
    abs_error = np.abs(error)

    return {
        "business_id": business_ids,
        "y_true": y_true,
        "y_pred": y_pred_out,
        "error": error,
        "abs_error": abs_error,
        "n_available": n_available,
        "n_used": n_used,
        "photo_count_col": photo_count_col,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def save_predictions_csv(out_path: Path, *, payload: dict[str, np.ndarray]) -> None:
    ensure_dir(out_path.parent)

    bid = payload["business_id"]
    y_true = payload["y_true"]
    y_pred = payload["y_pred"]
    err = payload["error"]
    abs_err = payload["abs_error"]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("business_id,y_true,y_pred,error,abs_error\n")
        for b, yt, yp, e, ae in zip(bid, y_true, y_pred, err, abs_err):
            f.write(f"{b},{yt:.4f},{yp:.4f},{e:.4f},{ae:.4f}\n")


def plot_true_vs_pred(
    out_path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    color_by: np.ndarray | None = None,
    color_label: str = "",
    title: str = "",
    xlim=(1, 5),
    ylim=(1, 5),
) -> None:
    plt.figure()

    if color_by is None:
        plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    else:
        vmax = int(max(1, np.max(color_by)))
        norm = mcolors.LogNorm(vmin=1, vmax=vmax) if vmax > 1 else None
        sc = plt.scatter(y_true, y_pred, c=np.clip(color_by, 1, None), s=8, alpha=0.5, norm=norm)
        cb = plt.colorbar(sc)
        cb.set_label(color_label)

    plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]])
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("True stars")
    plt.ylabel("Predicted stars")
    plt.title(title)
    plt.tight_layout()

    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_hist(out_path: Path, *, error: np.ndarray, title: str) -> None:
    plt.figure()
    plt.hist(error, bins=50)
    plt.xlabel("Prediction error (y_pred - y_true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()

    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_binned_calibration(out_path: Path, *, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    A lightweight calibration-style view:
    - bin true stars to 0.5 increments
    - plot avg predicted per bin
    """
    bins = np.round(y_true * 2) / 2.0
    uniq = np.unique(bins)
    means = np.array([float(np.mean(y_pred[bins == u])) for u in uniq], dtype=np.float32)

    plt.figure()
    plt.plot(uniq, means, marker="o")
    plt.plot([1, 5], [1, 5])
    plt.xlabel("True stars (binned to 0.5)")
    plt.ylabel("Avg predicted stars")
    plt.title(title)
    plt.tight_layout()

    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_all_plots(outdir: Path, *, payload: dict[str, np.ndarray], mae: float, rmse: float) -> None:
    y_true = payload["y_true"]
    y_pred = payload["y_pred"]
    error = payload["error"]
    n_available = payload["n_available"]

    plot_true_vs_pred(
        outdir / "scatter_true_vs_pred.png",
        y_true=y_true,
        y_pred=y_pred,
        title=f"True vs Predicted (val) | MAE={mae:.3f}, RMSE={rmse:.3f}",
    )

    plot_true_vs_pred(
        outdir / "scatter_true_vs_pred_colored_photos.png",
        y_true=y_true,
        y_pred=y_pred,
        color_by=n_available,
        color_label="Available photos per business (log scale)",
        title=f"True vs Predicted (val) colored by #photos | MAE={mae:.3f}, RMSE={rmse:.3f}",
    )

    plot_error_hist(
        outdir / "hist_error.png",
        error=error,
        title="Error distribution (val)",
    )

    plot_binned_calibration(
        outdir / "calibration_binned.png",
        y_true=y_true,
        y_pred=y_pred,
        title="Binned calibration (val)",
    )

    # Low-rating focus
    mask_low = y_true < 3.0
    if np.any(mask_low):
        plot_true_vs_pred(
            outdir / "scatter_lowstars_colored_photos.png",
            y_true=y_true[mask_low],
            y_pred=y_pred[mask_low],
            color_by=n_available[mask_low],
            color_label="Available photos (log scale)",
            title="Low-rating focus (val) colored by #photos",
            xlim=(1, 3),
            ylim=(1, 4.5),
        )
