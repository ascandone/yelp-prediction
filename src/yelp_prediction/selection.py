# src/yelp_prediction/rank_try_enhance.py
import argparse
import glob
import json
import os


def pick_metric(d: dict, k: str):
    # Tries common layouts used by try_enhance/sweep outputs
    for key in ["best_val", "best", "best_metrics", "best_metrics_val", "best_val_metrics"]:
        v = d.get(key)
        if isinstance(v, dict) and k in v:
            return v[k]
    v = d.get("best")
    if isinstance(v, dict) and k in v:
        return v[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="data/try_enhance/runs", help="Folder containing run_*.json files")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--sort-by", choices=["rmse", "mae"], default="rmse")
    args = ap.parse_args()

    pattern = os.path.join(args.runs_dir, "*.json")
    files = glob.glob(pattern)

    rows = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"[WARN] Could not read {f}: {e}")
            continue

        name = d.get("run_name") or d.get("name") or os.path.basename(f)
        mae = pick_metric(d, "mae")
        rmse = pick_metric(d, "rmse")

        if rmse is None and mae is None:
            continue

        rows.append((rmse, mae, name, f))

    # Filter out those without the sort metric
    if args.sort_by == "mae":
        rows = [r for r in rows if r[0] is not None]
        rows.sort(key=lambda x: (x[0], x[1] if x[1] is not None else 1e9))
    else:
        rows = [r for r in rows if r[1] is not None]
        rows.sort(key=lambda x: (x[1], x[0] if x[0] is not None else 1e9))

    print(f"TOTAL RUNS (with {args.sort_by}): {len(rows)}")
    if not rows:
        print("No valid runs found. Check --runs-dir.")
        return

    best = rows[0]
    print("\nBEST:")
    print(f"  sort_by={args.sort_by}")
    print(f"  rmse={best[0]}")
    print(f"  mae={best[1]}")
    print(f"  name={best[2]}")
    print(f"  file={best[3]}")

    print(f"\nTOP {args.topk}:")
    for r in rows[: args.topk]:
        rmse_s = f"{r[0]:.6f}" if isinstance(r[0], (int, float)) else str(r[0])
        mae_s = f"{r[1]:.6f}" if isinstance(r[1], (int, float)) else str(r[1])
        print(f"  rmse={rmse_s} mae={mae_s} | {r[2]}")


if __name__ == "__main__":
    main()
