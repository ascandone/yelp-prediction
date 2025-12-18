import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip")
    ap.add_argument("--preset", choices=["quick", "full"], default="quick")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--outdir", default="data/try_enhance")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_tag = f"sweep{args.preset}_{stamp}"
    outdir = Path(args.outdir)
    (outdir / "sweeps").mkdir(parents=True, exist_ok=True)

    # Grid definitions
    if args.preset == "quick":
        heads = [("mlp", [1024, 512])]
        acts = ["gelu", "relu"]
        dropouts = [0.2, 0.3]
        losses = ["mae", "mse"]
        opts = [("adamw", 0.01), ("adam", 0.0)]
        lrs = [5e-4, 1e-3]
        l2s = [False]  # we already saw l2norm is worse; keep off in quick
    else:
        # Bigger grid (still centered around CLIP head tuning)
        heads = [
            ("mlp", [1024, 512]),
            ("mlp", [512]),
            ("mlp", [2048, 512]),
            ("default", [1024, 512]),
            ("linear", [1024, 512]),
        ]
        acts = ["gelu", "relu"]
        dropouts = [0.1, 0.2, 0.3]
        losses = ["mae", "mse"]
        opts = [("adamw", 0.01), ("adamw", 0.0), ("adam", 0.0)]
        lrs = [1e-3, 5e-4, 1e-4]
        l2s = [False, True]

    runs = []
    idx = 0

    for (head, dims), act, do, loss, (opt, wd), lr, l2 in itertools.product(
        heads, acts, dropouts, losses, opts, lrs, l2s
    ):
        idx += 1
        run_name = f"{sweep_tag}__{idx:03d}"

        cmd = [
            sys.executable, "-u", "src/yelp_prediction/try_enhance.py",
            "--model", args.model,
            "--head", head,
            "--mlp-dims", *map(str, dims),
            "--dropout", str(do),
            "--act", act,
            "--optimizer", opt,
            "--weight-decay", str(wd),
            "--loss", loss,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(lr),
            "--split-seed", str(args.split_seed),
            "--run-name", run_name,
            "--tag", sweep_tag,
        ]
        if l2:
            cmd.append("--l2norm")
        if args.overwrite:
            cmd.append("--overwrite")

        print("\n[RUN]", run_name)
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        run_json = Path(args.outdir) / "runs" / f"run_{run_name}.json"
        if run_json.exists():
            with run_json.open("r", encoding="utf-8") as f:
                j = json.load(f)
            best = j.get("best_val", {})
            runs.append({
                "run_name": run_name,
                "head": head,
                "dims": "-".join(map(str, dims)),
                "act": act,
                "dropout": do,
                "loss": loss,
                "opt": opt,
                "wd": wd,
                "lr": lr,
                "l2norm": int(l2),
                "mae": best.get("mae"),
                "rmse": best.get("rmse"),
                "run_json": str(run_json).replace("\\", "/"),
            })

    # Save sweep summary
    summary_path = outdir / "sweeps" / f"{sweep_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    # Print best by RMSE
    runs_ok = [r for r in runs if r["rmse"] is not None]
    runs_ok.sort(key=lambda r: (r["rmse"], r["mae"]))
    if runs_ok:
        best = runs_ok[0]
        print("\n[BEST SWEEP RUN]")
        print(best)

    print("\n[SWEEP SUMMARY WRITTEN]", str(summary_path))


if __name__ == "__main__":
    main()
