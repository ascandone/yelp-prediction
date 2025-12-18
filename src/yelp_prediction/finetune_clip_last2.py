# src/yelp_prediction/finetune_clip_last2.py
# Fine-tune CLIP by unfreezing ONLY the last 2 vision transformer blocks (+ visual_projection),
# training on PHOTO-level samples with BUSINESS-level split (leakage-safe).
#
# Default head/hparams mirror the best sweep setting you found:
#   head=mlp dims=1024 512 act=gelu dropout=0.3 opt=adamw wd=0.01 loss=mse
#
# Typical Colab run:
#   poetry run python -u src/yelp_prediction/finetune_clip_last2.py --device cuda --epochs 5 --batch-size 64
#
# Optional: initialize head from your best frozen-embeddings checkpoint:
#   poetry run python -u src/yelp_prediction/finetune_clip_last2.py --device cuda \
#     --init-head-from data/try_enhance/models/<your_best_head>.pth

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from transformers import CLIPModel

import dataframes  # src/yelp_prediction/dataframes.py


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Split logic (same spirit as test_2.py / leakage-safe BUSINESS split)
# -------------------------
Row = Tuple[str, str, float]  # (photo_id, business_id, stars)


def split_rows(
    rows: List[Row],
    seed: int,
    mode: str = "business",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[Row], List[Row], List[Row]]:
    if mode == "photo":
        groups = sorted({pid for pid, _, _ in rows})
        key_fn = lambda r: r[0]
    else:
        groups = sorted({bid for _, bid, _ in rows})
        key_fn = lambda r: r[1]

    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train : n_train + n_val])
    test_g = set(groups[n_train + n_val :])

    train = [r for r in rows if key_fn(r) in train_g]
    val = [r for r in rows if key_fn(r) in val_g]
    test = [r for r in rows if key_fn(r) in test_g]
    return train, val, test


# -------------------------
# Dataset: load raw images by photo_id
# -------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    if train:
        aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.80, 1.00), ratio=(0.75, 1.3333)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
        return aug

    # eval
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class PhotoImageDataset(Dataset):
    def __init__(self, rows: Sequence[Row], photos_dir: Path, transform: transforms.Compose):
        self.rows = list(rows)
        self.photos_dir = photos_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        pid, bid, y = self.rows[i]
        path = self.photos_dir / f"{pid}.jpg"

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                x = self.transform(img)
        except Exception:
            # keep training robust; return a "blank" image tensor
            x = torch.zeros(3, 224, 224, dtype=torch.float32)

        return x, torch.tensor(float(y), dtype=torch.float32), pid, bid


# -------------------------
# Head (same idea as try_enhance: MLP + optional bounded output)
# -------------------------
def make_activation(name: str) -> nn.Module:
    n = (name or "relu").lower()
    if n == "gelu":
        return nn.GELU()
    if n in ("leakyrelu", "lrelu"):
        return nn.LeakyReLU(0.1)
    return nn.ReLU()


class HeadMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mlp_dims: Sequence[int],
        act: str = "gelu",
        dropout: float = 0.3,
        bounded: bool = True,
        init_center: float = 3.62,
    ):
        super().__init__()
        dims = [in_dim] + list(mlp_dims) + [1]
        layers: List[nn.Module] = []
        for j in range(len(dims) - 2):
            layers.append(nn.Linear(dims[j], dims[j + 1]))
            layers.append(make_activation(act))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        self.bounded = bool(bounded)
        self.init_center = float(init_center)

        # Mildly bias final layer to start near init_center when bounded
        if self.bounded:
            with torch.no_grad():
                last = None
                for m in reversed(self.net):
                    if isinstance(m, nn.Linear):
                        last = m
                        break
                if last is not None:
                    # y = 1 + 4*sigmoid(z) => z = logit((y-1)/4)
                    p = min(max((self.init_center - 1.0) / 4.0, 1e-4), 1.0 - 1e-4)
                    last.bias.fill_(math.log(p / (1.0 - p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1)
        if self.bounded:
            return 1.0 + 4.0 * torch.sigmoid(z)
        return z


# -------------------------
# CLIP + Head wrapper
# -------------------------
class ClipRegressor(nn.Module):
    def __init__(self, clip: CLIPModel, head: nn.Module, l2norm: bool = False):
        super().__init__()
        self.clip = clip
        self.head = head
        self.l2norm = bool(l2norm)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.clip.get_image_features(pixel_values=pixel_values)  # [B, D]
        if self.l2norm:
            feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        return self.head(feats)


def freeze_all_clip(clip: CLIPModel) -> None:
    for p in clip.parameters():
        p.requires_grad = False


def unfreeze_clip_last2(clip: CLIPModel) -> None:
    """
    Unfreeze last 2 transformer blocks of vision encoder (+ visual_projection).
    This maps the paper idea ("unfreeze last 2 layers") to CLIP ViT.
    """
    # vision encoder blocks
    layers = None
    if hasattr(clip, "vision_model") and hasattr(clip.vision_model, "encoder"):
        enc = clip.vision_model.encoder
        if hasattr(enc, "layers"):
            layers = enc.layers

    if layers is None or len(layers) < 2:
        raise RuntimeError("Could not locate CLIP vision encoder layers (unexpected CLIP structure).")

    for block in layers[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # Also unfreeze post layer norm if present
    if hasattr(clip.vision_model, "post_layernorm"):
        for p in clip.vision_model.post_layernorm.parameters():
            p.requires_grad = True

    # And unfreeze projection
    if hasattr(clip, "visual_projection") and clip.visual_projection is not None:
        for p in clip.visual_projection.parameters():
            p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def eval_loop(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[Tuple[str, str, float, float]]]:
    model.eval()
    ys: List[float] = []
    ps: List[float] = []
    rows_out: List[Tuple[str, str, float, float]] = []  # (photo_id, business_id, y, pred)

    for x, y, pid, bid in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)

        y_cpu = y.detach().cpu().numpy().tolist()
        p_cpu = pred.detach().cpu().numpy().tolist()

        ys.extend(y_cpu)
        ps.extend(p_cpu)

        for i in range(len(y_cpu)):
            rows_out.append((pid[i], bid[i], float(y_cpu[i]), float(p_cpu[i])))

    ys_np = np.asarray(ys, dtype=np.float64)
    ps_np = np.asarray(ps, dtype=np.float64)
    mae = float(np.mean(np.abs(ps_np - ys_np)))
    rmse = float(np.sqrt(np.mean((ps_np - ys_np) ** 2)))
    return mae, rmse, rows_out


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    loss_name: str,
    opt_name: str,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
    patience: int,
    select_best: str,
    amp: bool,
    out_dir: Path,
    run_name: str,
) -> Dict:
    model.to(device)

    # Loss
    loss_name = (loss_name or "mse").lower()
    if loss_name == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    # Optimizer param groups: backbone vs head
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for p in model.clip.parameters() if p.requires_grad]

    opt_name = (opt_name or "adamw").lower()
    if opt_name == "adam":
        opt_cls = torch.optim.Adam
    else:
        opt_cls = torch.optim.AdamW

    optimizer = opt_cls(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp and device.type == "cuda"))

    best_epoch = -1
    best = {"mae": float("inf"), "rmse": float("inf")}
    best_state = None
    bad = 0

    select_best = (select_best or "mae").lower()
    if select_best not in ("mae", "rmse"):
        select_best = "mae"

    history: List[Dict] = []

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        t0 = time.time()
        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(amp and device.type == "cuda")):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = int(x.shape[0])
            total_loss += float(loss.detach().cpu().item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)

        val_mae, val_rmse, _ = eval_loop(model, val_loader, device)

        dt = time.time() - t0
        print(
            f"[EPOCH {ep:02d}] train_loss={train_loss:.6f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} time={dt:.1f}s",
            flush=True,
        )

        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )

        score = val_mae if select_best == "mae" else val_rmse
        best_score = best["mae"] if select_best == "mae" else best["rmse"]

        if score < best_score - 1e-8:
            best_epoch = ep
            best = {"mae": val_mae, "rmse": val_rmse}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] patience={patience}", flush=True)
                break

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Save val preds for best
    val_mae, val_rmse, val_rows = eval_loop(model, val_loader, device)
    preds_path = out_dir / "preds" / f"val_preds_{run_name}.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "business_id", "y_true", "y_pred"])
        for pid, bid, y, p in val_rows:
            w.writerow([pid, bid, y, p])

    # Save model
    model_path = out_dir / "models" / f"best_{run_name}.pth"
    torch.save(model.state_dict(), model_path)

    # Save run json
    run_json = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val": {"mae": float(val_mae), "rmse": float(val_rmse)},
        "select_best": select_best,
        "history": history,
        "saved": {
            "model": str(model_path).replace("\\", "/"),
            "val_preds": str(preds_path).replace("\\", "/"),
        },
    }
    run_path = out_dir / "runs" / f"run_{run_name}.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2)

    print("[DONE]", flush=True)
    print("[BEST VAL]", run_json["best_val"], flush=True)
    print("[SAVED MODEL]", run_json["saved"]["model"], flush=True)
    print("[SAVED VAL PREDS]", run_json["saved"]["val_preds"], flush=True)
    print("[SAVED RUN JSON]", str(run_path).replace("\\", "/"), flush=True)

    return run_json


# -------------------------
# Data building using Alessandro's q_features (restaurants + exact_stars + photos_agg)
# -------------------------
def unroll_rows_from_qfeatures(
    df_features,
    *,
    max_photos_per_business: int = 0,
    seed: int = 42,
) -> List[Row]:
    rng = random.Random(seed)
    out: List[Row] = []

    df2 = df_features.select(["business_id", "stars", "photo_ids"])
    for r in df2.to_dicts():
        bid = str(r["business_id"])
        y = float(r["stars"])
        pids = r["photo_ids"] or []

        if max_photos_per_business and max_photos_per_business > 0 and len(pids) > max_photos_per_business:
            pids = rng.sample(list(pids), k=max_photos_per_business)

        for pid in pids:
            out.append((str(pid), bid, y))

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--model", default="clip", choices=["clip"])
    p.add_argument("--clip-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Use mixed precision (recommended on GPU).")

    # Data
    p.add_argument("--photos-dir", default=str(dataframes.PHOTOS_DIR / "photos"))
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--max-photos-per-business", type=int, default=0)

    # Head (defaults = best sweep-style)
    p.add_argument("--head", default="mlp", choices=["mlp"])
    p.add_argument("--mlp-dims", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--act", default="gelu")
    p.add_argument("--bounded", action="store_true", help="Bound predictions into [1,5] via 1+4*sigmoid(z).")
    p.add_argument("--no-bounded", dest="bounded", action="store_false")
    p.set_defaults(bounded=True)
    p.add_argument("--l2norm", action="store_true", help="L2-normalize CLIP embeddings before head (usually OFF).")

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--loss", default="mse", choices=["mse", "mae"])
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--select-best", default="mae", choices=["mae", "rmse"])

    # Optional init
    p.add_argument("--init-head-from", default="", help="Path to a head checkpoint from try_enhance (state_dict).")

    # Output
    p.add_argument("--out-dir", default="data/finetune_clip_last2")
    p.add_argument("--run-tag", default="", help="Optional tag appended to run name.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.split_seed)

    device = torch.device(args.device)

    # Load features dataframe (restaurants + exact_stars + photo_ids)
    df_features = dataframes.q_features.collect()
    n_biz = len(df_features)
    stars_med = float(df_features["stars"].median())
    stars_avg = float(df_features["stars"].mean())
    stars_std = float(df_features["stars"].std())

    print(f"[DATA] businesses={n_biz}", flush=True)
    print(f"[DATA] stars median={stars_med:.2f} avg={stars_avg:.2f} stdev={stars_std:.2f}", flush=True)

    rows = unroll_rows_from_qfeatures(
        df_features,
        max_photos_per_business=args.max_photos_per_business,
        seed=args.split_seed,
    )

    # Split by BUSINESS (leakage-safe)
    train_rows, val_rows, test_rows = split_rows(
        rows,
        seed=args.split_seed,
        mode="business",
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    print(f"[SPLIT] train businesses~={int(round(args.train_frac*n_biz))} val businesses~={int(round(args.val_frac*n_biz))} test businesses~={n_biz - int(round(args.train_frac*n_biz)) - int(round(args.val_frac*n_biz))}", flush=True)
    print(f"[SPLIT] train photos={len(train_rows)} val photos={len(val_rows)} test photos={len(test_rows)}", flush=True)

    photos_dir = Path(args.photos_dir)
    if not photos_dir.exists():
        raise FileNotFoundError(f"photos dir not found: {photos_dir}")

    train_tf = build_transforms(train=True, image_size=224)
    val_tf = build_transforms(train=False, image_size=224)

    train_ds = PhotoImageDataset(train_rows, photos_dir=photos_dir, transform=train_tf)
    val_ds = PhotoImageDataset(val_rows, photos_dir=photos_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Load CLIP
    clip = CLIPModel.from_pretrained(args.clip_name)
    freeze_all_clip(clip)
    unfreeze_clip_last2(clip)

    # Head
    head = HeadMLP(
        in_dim=512,  # CLIP ViT-B/32 image_features dim
        mlp_dims=args.mlp_dims,
        act=args.act,
        dropout=args.dropout,
        bounded=args.bounded,
        init_center=stars_avg,
    )

    # Optional init from best head checkpoint (try_enhance)
    if args.init_head_from:
        pth = Path(args.init_head_from)
        if not pth.exists():
            raise FileNotFoundError(f"--init-head-from not found: {pth}")

        sd = torch.load(pth, map_location="cpu")
        # The saved checkpoint may be:
        # - full model state_dict from try_enhance (just the head)
        # We'll try load directly. If keys mismatch, we'll attempt key-stripping.
        try:
            head.load_state_dict(sd, strict=True)
            print(f"[INIT] loaded head weights from: {pth}", flush=True)
        except Exception:
            # Try to extract sub-keys if it was saved as part of a bigger dict
            if isinstance(sd, dict):
                # best effort: filter keys that match head.*
                filtered = {k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}
                if filtered:
                    head.load_state_dict(filtered, strict=False)
                    print(f"[INIT] loaded partial head weights from: {pth}", flush=True)
                else:
                    raise

    model = ClipRegressor(clip=clip, head=head, l2norm=args.l2norm)

    n_trainable = count_trainable_params(model)
    print(f"[MODEL] clip={args.clip_name} unfreeze=last2 + projection", flush=True)
    print(f"[MODEL] head=mlp dims={args.mlp_dims} act={args.act} dropout={args.dropout} bounded={args.bounded} l2norm={args.l2norm}", flush=True)
    print(f"[MODEL] trainable_params={n_trainable:,}", flush=True)
    print(f"[TRAIN] loss={args.loss} opt={args.optimizer} lr_head={args.lr_head} lr_backbone={args.lr_backbone} wd={args.weight_decay} amp={bool(args.amp)} device={device.type}", flush=True)

    run_tag = f"__{args.run_tag}" if args.run_tag else ""
    run_name = (
        f"ftlast2_clip"
        f"__mlp__dims{'-'.join(map(str,args.mlp_dims))}"
        f"__act{args.act}"
        f"__drop{args.dropout}"
        f"__lrH{args.lr_head}"
        f"__lrB{args.lr_backbone}"
        f"__seed{args.split_seed}"
        f"{run_tag}"
    ).replace("/", "_").replace("\\", "_")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    cfg_path = out_dir / "runs" / f"config_{run_name}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["trainable_params"] = int(n_trainable)
    cfg["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Train + save best
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        loss_name=args.loss,
        opt_name=args.optimizer,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        patience=args.patience,
        select_best=args.select_best,
        amp=bool(args.amp),
        out_dir=out_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
