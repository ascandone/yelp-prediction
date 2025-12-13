import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import dataframes


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_edges(s: str) -> list[int]:
    edges = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    edges = sorted(set(edges))
    if len(edges) < 2:
        raise ValueError("Bucket edges must have at least 2 integers, e.g. 0,1,2,5,10")
    return edges


def _bucket_expr(x: pl.Expr, edges: list[int], name: str) -> pl.Expr:
    """
    Create categorical buckets without relying on Expr.cut (version-safe).
    Buckets are: [e0,e1), [e1,e2), ... [e_{n-2}, e_{n-1}) and a final ">= last".
    """
    expr = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"[{lo},{hi})"
        cond = (x >= lo) & (x < hi)
        if expr is None:
            expr = pl.when(cond).then(pl.lit(label))
        else:
            expr = expr.when(cond).then(pl.lit(label))
    expr = expr.otherwise(pl.lit(f">={edges[-1]}"))
    return expr.alias(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="reports/diagnostics")
    parser.add_argument("--pred_csv", type=str, default="reports/figures/val_predictions.csv")
    parser.add_argument("--low_thr", type=float, default=3.0)
    parser.add_argument("--restaurants_only", action="store_true", default=True)
    parser.add_argument("--photo_bucket_edges", type=str, default="0,1,2,3,5,10,20,50,100,1000000")
    parser.add_argument("--review_bucket_edges", type=str, default="0,1,5,10,20,50,100,200,500,1000,1000000")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # -------------------------
    # Paths (match dataframes.py)
    # -------------------------
    path_business = Path(dataframes.PATH_BUSINESS)
    path_review = Path(dataframes.PATH_REVIEWS)
    photos_json = Path(dataframes.PHOTOS_DIR) / "photos.json"

    for p in [path_business, path_review, photos_json]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    low_thr = float(args.low_thr)

    # -------------------------
    # Business base (optionally restrict to Restaurants to match training domain)
    # -------------------------
    q_biz = pl.scan_ndjson(path_business).select([
        pl.col("business_id"),
        pl.col("categories").cast(pl.Utf8).fill_null("").alias("categories"),
        pl.col("stars").cast(pl.Float32).alias("stars_business"),
        pl.col("review_count").cast(pl.Int32).alias("review_count_business"),
    ])

    if args.restaurants_only:
        q_biz = q_biz.filter(pl.col("categories").str.contains("Restaurants"))

    business = q_biz.collect()
    restaurant_ids_lf = business.select("business_id").lazy()

    # -------------------------
    # Photos aggregation (per business) - filtered to same business universe
    # -------------------------
    cap = pl.col("caption").cast(pl.Utf8).fill_null("").str.strip_chars()

    photos_agg = (
        pl.scan_ndjson(photos_json)
        .select([
            pl.col("business_id"),
            pl.col("photo_id"),
            pl.col("caption").cast(pl.Utf8),
        ])
        .join(restaurant_ids_lf, on="business_id", how="inner")
        .with_columns([
            cap.str.len_chars().alias("caption_len"),
            (cap.str.len_chars() == 0).alias("caption_empty"),
        ])
        .group_by("business_id")
        .agg([
            pl.len().alias("photo_count"),
            pl.sum("caption_empty").cast(pl.Int32).alias("photo_caption_empty_n"),
            pl.mean("caption_len").cast(pl.Float32).alias("photo_caption_len_mean"),
        ])
        .collect()
    )

    # -------------------------
    # Reviews aggregation (per business) - include stars mean + text emptiness
    # -------------------------
    txt = pl.col("text").cast(pl.Utf8).fill_null("").str.strip_chars()

    reviews_agg = (
        pl.scan_ndjson(path_review)
        .select([
            pl.col("business_id"),
            pl.col("stars").cast(pl.Float32).alias("stars_review"),
            pl.col("text").cast(pl.Utf8),
        ])
        .join(restaurant_ids_lf, on="business_id", how="inner")
        .with_columns([
            txt.str.len_chars().alias("text_len"),
            (txt.str.len_chars() == 0).alias("text_empty"),
        ])
        .group_by("business_id")
        .agg([
            pl.len().alias("reviews_n"),
            pl.mean("stars_review").alias("stars_review_mean"),
            pl.sum("text_empty").cast(pl.Int32).alias("reviews_text_empty_n"),
            (pl.len() - pl.sum("text_empty")).cast(pl.Int32).alias("reviews_text_nonempty_n"),
            pl.mean("text_len").cast(pl.Float32).alias("review_text_len_mean"),
        ])
        .collect()
    )

    # -------------------------
    # Merge + derived flags
    # -------------------------
    df = (
        business
        .join(photos_agg, on="business_id", how="left")
        .join(reviews_agg, on="business_id", how="left")
        .with_columns([
            pl.col("photo_count").fill_null(0).cast(pl.Int32),
            pl.col("reviews_n").fill_null(0).cast(pl.Int32),
            pl.col("reviews_text_empty_n").fill_null(0).cast(pl.Int32),
            pl.col("reviews_text_nonempty_n").fill_null(0).cast(pl.Int32),

            (pl.col("photo_count") > 0).alias("has_photos"),
            (pl.col("reviews_n") > 0).alias("has_reviews"),
            (pl.col("reviews_text_nonempty_n") > 0).alias("has_text_reviews"),

            # ratio diagnostics
            (pl.col("reviews_text_empty_n") / pl.when(pl.col("reviews_n") == 0).then(None).otherwise(pl.col("reviews_n"))).alias("share_reviews_empty_text"),

            # prefer review-derived mean stars when available; fallback to business stars otherwise
            pl.coalesce([pl.col("stars_review_mean"), pl.col("stars_business")]).alias("stars_target"),
        ])
    )

    # -------------------------
    # Coverage summary
    # -------------------------
    total_business = df.height

    def pct(x: int) -> float:
        return 100.0 * x / max(1, total_business)

    n_has_photos = df.filter(pl.col("has_photos")).height
    n_no_photos = total_business - n_has_photos

    n_has_reviews = df.filter(pl.col("has_reviews")).height
    n_no_reviews = total_business - n_has_reviews

    n_has_text = df.filter(pl.col("has_text_reviews")).height
    n_no_text = total_business - n_has_text

    n_reviews_no_photos = df.filter(pl.col("has_reviews") & (~pl.col("has_photos"))).height
    n_no_photos_no_text = df.filter((~pl.col("has_photos")) & (~pl.col("has_text_reviews"))).height

    n_low_all = df.filter(pl.col("stars_target") < low_thr).height
    n_low_with_photos = df.filter(pl.col("has_photos") & (pl.col("stars_target") < low_thr)).height
    n_low_no_photos = df.filter((~pl.col("has_photos")) & (pl.col("stars_target") < low_thr)).height

    summary_txt = outdir / "coverage_summary.txt"
    summary_txt.write_text(
        "\n".join([
            f"UNIVERSE: {'Restaurants only' if args.restaurants_only else 'All businesses'}",
            f"TOTAL BUSINESSES: {total_business}",

            "",
            f"BUSINESSES WITH PHOTOS: {n_has_photos} ({pct(n_has_photos):.2f}%)",
            f"BUSINESSES WITHOUT PHOTOS: {n_no_photos} ({pct(n_no_photos):.2f}%)",

            "",
            f"BUSINESSES WITH REVIEWS (from review.json): {n_has_reviews} ({pct(n_has_reviews):.2f}%)",
            f"BUSINESSES WITHOUT REVIEWS (from review.json): {n_no_reviews} ({pct(n_no_reviews):.2f}%)",

            "",
            f"BUSINESSES WITH NON-EMPTY TEXT REVIEWS: {n_has_text} ({pct(n_has_text):.2f}%)",
            f"BUSINESSES WITH NO NON-EMPTY TEXT REVIEWS: {n_no_text} ({pct(n_no_text):.2f}%)",

            "",
            f"BUSINESSES WITH REVIEWS BUT NO PHOTOS: {n_reviews_no_photos} ({pct(n_reviews_no_photos):.2f}%)",
            f"BUSINESSES WITH NO PHOTOS AND NO TEXT REVIEWS: {n_no_photos_no_text} ({pct(n_no_photos_no_text):.2f}%)",

            "",
            f"LOW-RATING THRESHOLD: stars < {low_thr}",
            f"LOW-RATING (ALL BUSINESSES): {n_low_all} ({pct(n_low_all):.2f}%)",
            f"LOW-RATING (WITH PHOTOS): {n_low_with_photos} ({100.0*n_low_with_photos/max(1,n_has_photos):.2f}% of with-photos)",
            f"LOW-RATING (WITHOUT PHOTOS): {n_low_no_photos} ({100.0*n_low_no_photos/max(1,n_no_photos):.2f}% of no-photos)",
        ]) + "\n",
        encoding="utf-8"
    )

    # Save per-business coverage table
    coverage_csv = outdir / "business_coverage.csv"
    df.write_csv(coverage_csv)

    # -------------------------
    # Plots: stars distribution and stars vs counts
    # -------------------------
    stars_all = df["stars_target"].to_numpy()
    stars_with = df.filter(pl.col("has_photos"))["stars_target"].to_numpy()
    stars_without = df.filter(~pl.col("has_photos"))["stars_target"].to_numpy()

    plt.figure()
    plt.hist(stars_all, bins=40, alpha=0.6, label="All")
    plt.hist(stars_with, bins=40, alpha=0.6, label="With photos")
    plt.hist(stars_without, bins=40, alpha=0.6, label="Without photos")
    plt.xlabel("Stars (stars_target)")
    plt.ylabel("Count")
    plt.title("Stars distribution: all vs with photos vs without photos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "stars_distribution_by_photo_presence.png", dpi=200)
    plt.close()

    plt.figure()
    x = df["photo_count"].to_numpy()
    y = df["stars_target"].to_numpy()
    plt.scatter(x + 1, y, s=6, alpha=0.3)
    plt.xscale("log")
    plt.xlabel("photo_count + 1 (log scale)")
    plt.ylabel("stars_target")
    plt.title("Stars vs photo_count")
    plt.tight_layout()
    plt.savefig(outdir / "stars_vs_photo_count.png", dpi=200)
    plt.close()

    plt.figure()
    x = df["reviews_n"].to_numpy()
    y = df["stars_target"].to_numpy()
    plt.scatter(x + 1, y, s=6, alpha=0.3)
    plt.xscale("log")
    plt.xlabel("reviews_n + 1 (log scale)")
    plt.ylabel("stars_target")
    plt.title("Stars vs number of reviews")
    plt.tight_layout()
    plt.savefig(outdir / "stars_vs_review_count.png", dpi=200)
    plt.close()

    # -------------------------
    # Optional: merge with predictions to analyze error vs coverage
    # -------------------------
    pred_path = Path(args.pred_csv)
    if pred_path.exists():
        preds = pl.read_csv(pred_path).with_columns([
            pl.col("business_id").cast(pl.Utf8)
        ])
        preds = preds.select(["business_id", "y_true", "y_pred", "error", "abs_error"])

        merged = preds.join(
            df.select([
                "business_id",
                "photo_count",
                "reviews_n",
                "reviews_text_nonempty_n",
                "reviews_text_empty_n",
                "has_photos",
                "has_reviews",
                "has_text_reviews",
                "stars_target",
            ]),
            on="business_id",
            how="left",
        )

        merged_csv = outdir / "preds_joined_coverage.csv"
        merged.write_csv(merged_csv)

        plt.figure()
        x = merged["photo_count"].fill_null(0).to_numpy()
        y = merged["abs_error"].to_numpy()
        plt.scatter(x + 1, y, s=8, alpha=0.4)
        plt.xscale("log")
        plt.xlabel("photo_count + 1 (log scale)")
        plt.ylabel("abs_error")
        plt.title("Abs error vs photo_count")
        plt.tight_layout()
        plt.savefig(outdir / "abs_error_vs_photo_count.png", dpi=200)
        plt.close()

        plt.figure()
        x = merged["reviews_n"].fill_null(0).to_numpy()
        y = merged["abs_error"].to_numpy()
        plt.scatter(x + 1, y, s=8, alpha=0.4)
        plt.xscale("log")
        plt.xlabel("reviews_n + 1 (log scale)")
        plt.ylabel("abs_error")
        plt.title("Abs error vs number of reviews")
        plt.tight_layout()
        plt.savefig(outdir / "abs_error_vs_review_count.png", dpi=200)
        plt.close()

        # MAE by photo buckets (version-safe)
        photo_edges = _parse_edges(args.photo_bucket_edges)
        merged2 = merged.with_columns([
            pl.col("photo_count").fill_null(0).cast(pl.Int32).alias("photo_count_i"),
            _bucket_expr(pl.col("photo_count").fill_null(0).cast(pl.Int32), photo_edges, "photo_bucket"),
        ])
        mae_by_bucket = (
            merged2.group_by("photo_bucket")
            .agg([
                pl.len().alias("n"),
                pl.col("abs_error").mean().alias("mae"),
            ])
            .sort("photo_bucket")
        )
        mae_by_bucket.write_csv(outdir / "mae_by_photo_bucket.csv")

        # MAE by review buckets (version-safe)
        review_edges = _parse_edges(args.review_bucket_edges)
        merged3 = merged.with_columns([
            pl.col("reviews_n").fill_null(0).cast(pl.Int32).alias("reviews_n_i"),
            _bucket_expr(pl.col("reviews_n").fill_null(0).cast(pl.Int32), review_edges, "review_bucket"),
        ])
        mae_by_rbucket = (
            merged3.group_by("review_bucket")
            .agg([
                pl.len().alias("n"),
                pl.col("abs_error").mean().alias("mae"),
            ])
            .sort("review_bucket")
        )
        mae_by_rbucket.write_csv(outdir / "mae_by_review_bucket.csv")

    print("OK")
    print(f"- Wrote: {summary_txt}")
    print(f"- Wrote: {coverage_csv}")
    print(f"- Plots in: {outdir}")


if __name__ == "__main__":
    main()
