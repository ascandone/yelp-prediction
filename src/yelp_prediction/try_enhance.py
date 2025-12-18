# src/yelp_prediction/try_enhance.py
# Tune "head / hyperparams" on top of precomputed embeddings (CPU-friendly).
# Aggregation logic:
# - split at BUSINESS level (to avoid leakage)
# - train/eval at PHOTO level (each photo gets the business stars as target)

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import dataframes  # local module: src/yelp_prediction/dataframes.py provides q_features


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_in_dim(emb: Dict[str, torch.Tensor]) -> int:
    k = next(iter(emb))
    v = emb[k]
    if isinstance(v, torch.Tensor):
        return int(v.shape[-1])
    raise TypeError(f"Embeddings dict must contain torch.Tensor values. Got: {type(v)}")


def resolve_features_path(model_tag: str, user_path: str) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"--features not found: {p}")
        return p

    p = Path(f"data/features/features-{model_tag}.pt")
    if p.exists():
        return p

    # fallback search
    hits = list(Path(".").glob(f"**/features-{model_tag}.pt"))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not locate features-{model_tag}.pt. Put it under data/features/ or pass --features."
    )


# -------------------------
# Dataset: unroll businesses -> photos
# -------------------------
Row = Tuple[str, str, float]  # (photo_id, business_id, stars)


class PhotoRowsDataset(Dataset):
    def __init__(self, rows: Sequence[Row], emb: Dict[str, torch.Tensor], *, l2norm: bool = False):
        self.rows = list(rows)
        self.emb = emb
        self.l2norm = bool(l2norm)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        pid, bid, y = self.rows[i]
        x = self.emb[pid].float()
        if self.l2norm:
            x = x / (x.norm(p=2) + 1e-12)
        return x, torch.tensor(y, dtype=torch.float32), pid, bid


def business_split(df, split_seed: int, train_frac: float):
    # Reproducible business-level split
    rng = np.random.default_rng(split_seed)
    mask = rng.random(len(df)) < train_frac
    train_df = df.filter(mask)
    val_df = df.filter(~mask)
    return train_df, val_df


def unroll_rows(
    df,
    emb: Dict[str, torch.Tensor],
    *,
    max_photos_per_business: int = 0,
    seed: int = 42,
) -> List[Row]:
    rng = random.Random(seed)
    out: List[Row] = []

    df2 = df.select(["business_id", "stars", "photo_ids"])
    for r in df2.to_dicts():
        bid = r["business_id"]
        stars = float(r["stars"])
        pids = list(r["photo_ids"])

        # keep only photos that exist in embedding dict
        pids = [pid for pid in pids if pid in emb]
        if not pids:
            continue

        if max_photos_per_business and len(pids) > max_photos_per_business:
            rng.shuffle(pids)
            pids = pids[:max_photos_per_business]

        out.extend([(pid, bid, stars) for pid in pids])

    return out


# -------------------------
# Model: configurable head + optional bounding to [1,5]
# -------------------------
def make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    return nn.ReLU()


def build_mlp(in_dim: int, dims: Sequence[int], act: str, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in dims:
        layers.append(nn.Linear(prev, h))
        layers.append(make_activation(act))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class EnhancedHeadModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        head: str,
        mlp_dims: Sequence[int],
        act: str,
        dropout: float,
        init_center: float,
        bounded: bool,
    ):
        super().__init__()
        head = (head or "default").lower()

        if head == "linear":
            self.head = nn.Linear(input_dim, 1)
        elif head == "default":
            # matches the "default" spirit: 512 + ReLU + Dropout + Linear
            self.head = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout if dropout > 0 else 0.2),
                nn.Linear(512, 1),
            )
        else:
            # "mlp" / paper-like head
            self.head = build_mlp(input_dim, mlp_dims, act=act, dropout=dropout)

        self.bounded = bool(bounded)

        # Initialize final bias so initial prediction ~ init_center (avg stars)
        init_center = float(init_center)
        init_center = min(5.0, max(1.0, init_center))
        p = (init_center - 1.0) / 4.0  # [0,1]
        p = min(0.999, max(0.001, p))
        initial_bias = torch.logit(torch.tensor(p))

        last = self.head[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.constant_(last.bias, initial_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(x).squeeze(-1)
        if self.bounded:
            return torch.sigmoid(raw) * 4.0 + 1.0
        return raw


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    abs_err = []
    sq_err = []
    outputs: List[Tuple[str, str, float]] = []  # (photo_id, business_id, pred)

    for x, y, pids, bids in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)

        diff = p - y
        abs_err.append(torch.abs(diff).detach().cpu())
        sq_err.append((diff ** 2).detach().cpu())

        p_cpu = p.detach().cpu().numpy().tolist()
        for pid, bid, pred in zip(list(pids), list(bids), p_cpu):
            outputs.append((pid, bid, float(pred)))

    if not abs_err:
        return {"mae": float("nan"), "rmse": float("nan")}, outputs

    abs_all = torch.cat(abs_err)
    sq_all = torch.cat(sq_err)
    mae = float(abs_all.mean().item())
    rmse = float(torch.sqrt(sq_all.mean()).item())
    return {"mae": mae, "rmse": rmse}, outputs


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    optimizer_name: str,
    patience: int,
    scheduler_name: str,
):
    model.to(device)

    loss_name = (loss_name or "mae").lower()
    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "huber":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        loss_fn = nn.L1Loss()  # mae-like

    optimizer_name = (optimizer_name or "adam").lower()
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    scheduler_name = (scheduler_name or "none").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)

    best_rmse = math.inf
    best_state = None
    best_val_outputs: List[Tuple[str, str, float]] = []
    best_metrics = {"mae": math.inf, "rmse": math.inf}
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            p = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += float(loss.item()) * bs
            n += bs

        if scheduler is not None:
            scheduler.step()

        train_loss = total / max(1, n)
        val_metrics, val_outputs = eval_metrics(model, val_loader, device)

        print(
            f"[EPOCH {ep:02d}] train_loss={train_loss:.6f}  "
            f"val_mae={val_metrics['mae']:.4f}  val_rmse={val_metrics['rmse']:.4f}"
        )

        # select best by RMSE (stable)
        if val_metrics["rmse"] < best_rmse - 1e-6:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            best_val_outputs = val_outputs
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] patience={patience}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics, best_val_outputs


# -------------------------
# Main
# -------------------------
@dataclass
class RunConfig:
    model_tag: str
    features_path: str
    head: str
    mlp_dims: List[int]
    act: str
    dropout: float
    bounded: bool
    l2norm: bool
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    loss: str
    optimizer: str
    scheduler: str
    patience: int
    split_seed: int
    train_frac: float
    max_photos_per_business: int
    num_workers: int
    device: str
    tag: str
    run_name: str


def _fmt_float(x: float) -> str:
    # file-friendly float string
    return f"{x:g}".replace(".", "p")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", dest="model_tag", default="clip")
    ap.add_argument("--features", default="")

    ap.add_argument("--head", default="default", choices=["default", "linear", "mlp"])
    ap.add_argument("--mlp-dims", nargs="*", type=int, default=[1024, 512])
    ap.add_argument("--act", default="relu", choices=["relu", "gelu", "leakyrelu"])
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--bounded", action="store_true", help="Constrain preds to [1,5] via sigmoid*4+1")
    ap.add_argument("--no-bounded", dest="bounded", action="store_false")
    ap.set_defaults(bounded=True)

    ap.add_argument("--l2norm", action="store_true")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--loss", default="mae", choices=["mae", "mse", "huber"])
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    ap.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    ap.add_argument("--patience", type=int, default=5)

    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--max-photos-per-business", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--outdir", default="data/try_enhance")

    # NEW: run naming and overwrite behavior
    ap.add_argument("--run-name", default="", help="If set, forces the run name (used for filenames).")
    ap.add_argument("--tag", default="", help="Optional suffix appended to auto run name.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing run artifacts for the same run-name.")

    args = ap.parse_args()
    set_seed(args.split_seed)

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "preds").mkdir(parents=True, exist_ok=True)
    (outdir / "runs").mkdir(parents=True, exist_ok=True)

    # Build deterministic run name (NO collisions)
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = (
            f"{args.model_tag}"
            f"__head{args.head}"
            f"__dims{'-'.join(map(str, args.mlp_dims))}"
            f"__act{args.act}"
            f"__do{_fmt_float(args.dropout)}"
            f"__b{int(args.bounded)}"
            f"__loss{args.loss}"
            f"__opt{args.optimizer}"
            f"__wd{_fmt_float(args.weight_decay)}"
            f"__lr{_fmt_float(args.lr)}"
            f"__l2{int(args.l2norm)}"
            f"__seed{args.split_seed}"
            f"__tf{_fmt_float(args.train_frac)}"
            f"__mpb{args.max_photos_per_business}"
        )
        if args.scheduler and args.scheduler != "none":
            run_name += f"__sch{args.scheduler}"
        run_name += f"__pat{args.patience}"

        if args.tag:
            run_name = f"{run_name}__{args.tag}"

    run_json = outdir / "runs" / f"run_{run_name}.json"
    if run_json.exists() and not args.overwrite:
        print(f"[SKIP] Run already exists: {run_json}")
        with run_json.open("r", encoding="utf-8") as f:
            old = json.load(f)
        print("[BEST VAL]", old.get("best_val"))
        print("[MODEL PATH]", old.get("model_path"))
        print("[PRED PATH]", old.get("pred_path"))
        return

    # Load embeddings
    features_path = resolve_features_path(args.model_tag, args.features)
    print("[LOAD] features:", str(features_path))
    emb = torch.load(features_path, map_location="cpu")
    in_dim = infer_in_dim(emb)
    print("[INFO] embeddings:", len(emb), "dim:", in_dim)

    # Collect features dataframe (IMPORTANT: deterministic ordering)
    df = dataframes.q_features.collect().sort("business_id")

    median = float(df["stars"].median())
    baseline_mae = float((df["stars"] - median).abs().mean())
    avg = float(df["stars"].mean())
    stdev = float(df["stars"].std())

    print("\n[DATA] businesses:", len(df))
    print(f"[DATA] stars median={median:.2f} baseline_mae={baseline_mae:.2f} avg={avg:.2f} stdev={stdev:.2f}")

    train_df, val_df = business_split(df, split_seed=args.split_seed, train_frac=args.train_frac)

    train_rows = unroll_rows(
        train_df,
        emb,
        max_photos_per_business=args.max_photos_per_business,
        seed=args.split_seed,
    )
    val_rows = unroll_rows(
        val_df,
        emb,
        max_photos_per_business=args.max_photos_per_business,
        seed=args.split_seed + 1,
    )

    if not train_rows or not val_rows:
        raise RuntimeError("No training/validation rows after unrolling. Check q_features and features alignment.")

    print(f"[SPLIT] train businesses={len(train_df)} val businesses={len(val_df)}")
    print(f"[SPLIT] train photos={len(train_rows)} val photos={len(val_rows)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("[DEVICE]", str(device))

    train_ds = PhotoRowsDataset(train_rows, emb, l2norm=args.l2norm)
    val_ds = PhotoRowsDataset(val_rows, emb, l2norm=args.l2norm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EnhancedHeadModel(
        input_dim=in_dim,
        head=args.head,
        mlp_dims=args.mlp_dims,
        act=args.act,
        dropout=args.dropout,
        init_center=avg,
        bounded=args.bounded,
    )

    print(f"\n[RUN NAME] {run_name}")
    print(f"[MODEL] head={args.head} mlp_dims={args.mlp_dims} act={args.act} dropout={args.dropout} bounded={args.bounded}")
    print(f"[TRAIN] loss={args.loss} opt={args.optimizer} lr={args.lr} wd={args.weight_decay} sched={args.scheduler} patience={args.patience}")
    print(f"[FLAGS] l2norm={args.l2norm} train_frac={args.train_frac} max_photos_per_business={args.max_photos_per_business} split_seed={args.split_seed}")

    trained, best_metrics, best_val_outputs = train_loop(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        patience=args.patience,
    )

    # Save model + preds + run json
    model_path = outdir / "models" / f"best_{run_name}.pth"
    pred_path = outdir / "preds" / f"val_preds_{run_name}.csv"

    torch.save(trained.state_dict(), model_path)

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo_id", "business_id", "pred"])
        for pid, bid, pred in best_val_outputs:
            w.writerow([pid, bid, f"{pred:.6f}"])

    cfg = RunConfig(
        model_tag=args.model_tag,
        features_path=str(features_path),
        head=args.head,
        mlp_dims=list(args.mlp_dims),
        act=args.act,
        dropout=float(args.dropout),
        bounded=bool(args.bounded),
        l2norm=bool(args.l2norm),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        loss=str(args.loss),
        optimizer=str(args.optimizer),
        scheduler=str(args.scheduler),
        patience=int(args.patience),
        split_seed=int(args.split_seed),
        train_frac=float(args.train_frac),
        max_photos_per_business=int(args.max_photos_per_business),
        num_workers=int(args.num_workers),
        device=str(device),
        tag=str(args.tag),
        run_name=str(run_name),
    )

    with run_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "best_val": best_metrics,
                "model_path": str(model_path),
                "pred_path": str(pred_path),
            },
            f,
            indent=2,
        )

    print("\n[DONE]")
    print("[BEST VAL]", best_metrics)
    print("[SAVED MODEL]", str(model_path))
    print("[SAVED VAL PREDS]", str(pred_path))
    print("[SAVED RUN JSON]", str(run_json))


if __name__ == "__main__":
    main()
