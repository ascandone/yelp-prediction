import argparse
from pathlib import Path

# Support both:
# - package imports:   python -m yelp_prediction.diagnose_bias
# - direct execution:  python src/yelp_prediction/diagnose_bias.py
try:  # pragma: no cover
    from yelp_prediction import bias_utils as B
except ImportError:  # pragma: no cover
    import bias_utils as B  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="reports/diagnostics")
    parser.add_argument("--pred_csv", type=str, default="reports/figures/val_predictions.csv")
    parser.add_argument("--low_thr", type=float, default=3.0)
    parser.add_argument("--restaurants_only", action="store_true", default=True)
    parser.add_argument("--photo_bucket_edges", type=str, default="0,1,2,3,5,10,20,50,100,1000000")
    parser.add_argument("--review_bucket_edges", type=str, default="0,1,5,10,20,50,100,200,500,1000,1000000")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    B.ensure_dir(outdir)

    df = B.build_coverage_df(restaurants_only=args.restaurants_only, low_thr=float(args.low_thr))

    # Base outputs
    summary_txt = outdir / "coverage_summary.txt"
    B.write_coverage_summary(summary_txt, df, restaurants_only=args.restaurants_only, low_thr=float(args.low_thr))

    coverage_csv = outdir / "business_coverage.csv"
    df.write_csv(coverage_csv)

    B.plot_basic_distributions(outdir, df)

    # Optional outputs if predictions exist
    pred_path = Path(args.pred_csv)
    if pred_path.exists():
        merged = B.merge_with_predictions(df, pred_path)

        merged_csv = outdir / "preds_joined_coverage.csv"
        merged.write_csv(merged_csv)

        B.plot_abs_error_vs_counts(outdir, merged)

        photo_edges = B.parse_edges(args.photo_bucket_edges)
        B.mae_by_bucket(
            outdir / "mae_by_photo_bucket.csv",
            merged,
            col="photo_count",
            edges=photo_edges,
            bucket_name="photo_bucket",
        )

        review_edges = B.parse_edges(args.review_bucket_edges)
        B.mae_by_bucket(
            outdir / "mae_by_review_bucket.csv",
            merged,
            col="reviews_n",
            edges=review_edges,
            bucket_name="review_bucket",
        )

    print("OK")
    print(f"- Wrote: {summary_txt}")
    print(f"- Wrote: {coverage_csv}")
    print(f"- Plots in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
