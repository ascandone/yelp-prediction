import argparse
from pathlib import Path

# Support both:
# - package imports:   python -m yelp_prediction.plot_predictions
# - direct execution:  python src/yelp_prediction/plot_predictions.py
try:  # pragma: no cover
    from yelp_prediction import eval_utils as E
except ImportError:  # pragma: no cover
    import eval_utils as E  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/features/features.pt")
    parser.add_argument("--model", type=str, default="best_mil_model.pth")
    parser.add_argument("--outdir", type=str, default="reports/figures")
    parser.add_argument("--max_photos", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = E.EvalConfig(
        features_path=Path(args.features),
        model_path=Path(args.model),
        outdir=Path(args.outdir),
        max_photos=args.max_photos,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    E.ensure_dir(cfg.outdir)

    df = E.load_features_df()
    _, val_df = E.make_train_val_split(df, seed=cfg.seed, train_frac=cfg.train_frac)

    median = E.compute_median_stars(df)
    features_dict = E.load_features_dict(cfg.features_path)
    model = E.load_model(cfg.model_path, median_stars=median)

    payload = E.predict_val(
        val_df,
        features_dict=features_dict,
        model=model,
        max_photos=cfg.max_photos,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        sampling="random",  # preserve current behavior
    )

    mae, rmse = E.compute_metrics(payload["y_true"], payload["y_pred"])

    pred_csv = cfg.outdir / "val_predictions.csv"
    E.save_predictions_csv(pred_csv, payload=payload)

    print(f"Validation size: {len(payload['y_true'])}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved: {pred_csv}")

    E.generate_all_plots(cfg.outdir, payload=payload, mae=mae, rmse=rmse)
    print(f"Saved figures to: {cfg.outdir.resolve()}")


if __name__ == "__main__":
    main()
