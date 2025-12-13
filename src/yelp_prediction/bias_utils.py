from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Support both:
# - package imports:   from yelp_prediction import bias_utils
# - direct execution:  python src/yelp_prediction/diagnose_bias.py
try:  # pragma: no cover
    from . import dataframes
except ImportError:  # pragma: no cover
    import dataframes  # type: ignore


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_edges(s: str) -> list[int]:
    edges = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    edges = sorted(set(edges))
    if len(edges) < 2:
        raise ValueError("Bucket edges must contain at least two integers, e.g. 0,1,2,5,10")
    return edges


def bucket_expr(x: pl.Expr, edges: list[int], name: str) -> pl.Expr:
    """
    Create version-safe categorical buckets without relying on Expr.cut.

    Buckets are: [e0,e1), [e1,e2), ... [e_{n-2}, e_{n-1}), plus a final '>= last'.
    """
    expr = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"[{lo},{hi})"
        cond = (x >= lo) & (x < hi)
        expr = pl.when(cond).then(pl.lit(label)) if expr is None else expr.when(cond).then(pl.lit(label))
    expr = expr.otherwise(pl.lit(f">={edges[-1]}"))
    return expr.alias(name)


def build_universe(*, restaurants_only: bool = True) -> pl.DataFrame:
    """
    Business-level universe for coverage analysis.

    By default, we keep only 'Restaurants' to match the modeling domain.
    """
    q_biz = dataframes.q_businesses.select(
        [
            pl.col("business_id"),
            pl.col("categories").cast(pl.Utf8).fill_null("").alias("categories"),
            pl.col("stars").cast(pl.Float32).alias("stars_business"),
            pl.col("review_count").cast(pl.Int32).alias("review_count_business"),
        ]
    )
    if restaurants_only:
        q_biz = q_biz.filter(pl.col("categories").str.contains("Restaurants"))
    return q_biz.collect()


def aggregate_photos(restaurant_ids_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Aggregate photo info per business_id for the selected universe.
    """
    cap = pl.col("caption").cast(pl.Utf8).fill_null("").str.strip_chars()

    return (
        dataframes.q_photos.select([pl.col("business_id"), pl.col("photo_id"), pl.col("caption").cast(pl.Utf8)])
        .join(restaurant_ids_lf, on="business_id", how="inner")
        .with_columns(
            [
                cap.str.len_chars().alias("caption_len"),
                (cap.str.len_chars() == 0).alias("caption_empty"),
            ]
        )
        .group_by("business_id")
        .agg(
            [
                pl.len().alias("photo_count"),
                pl.sum("caption_empty").cast(pl.Int32).alias("photo_caption_empty_n"),
                pl.mean("caption_len").cast(pl.Float32).alias("photo_caption_len_mean"),
            ]
        )
        .collect()
    )


def aggregate_reviews(restaurant_ids_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Aggregate review info per business_id for the selected universe.

    We explicitly measure the fraction of reviews with empty text.
    """
    txt = pl.col("text").cast(pl.Utf8).fill_null("").str.strip_chars()

    return (
        dataframes.q_reviews.select(
            [
                pl.col("business_id"),
                pl.col("stars").cast(pl.Float32).alias("stars_review"),
                pl.col("text").cast(pl.Utf8),
            ]
        )
        .join(restaurant_ids_lf, on="business_id", how="inner")
        .with_columns(
            [
                txt.str.len_chars().alias("text_len"),
                (txt.str.len_chars() == 0).alias("text_empty"),
            ]
        )
        .group_by("business_id")
        .agg(
            [
                pl.len().alias("reviews_n"),
                pl.mean("stars_review").alias("stars_review_mean"),
                pl.sum("text_empty").cast(pl.Int32).alias("reviews_text_empty_n"),
                (pl.len() - pl.sum("text_empty")).cast(pl.Int32).alias("reviews_text_nonempty_n"),
                pl.mean("text_len").cast(pl.Float32).alias("review_text_len_mean"),
            ]
        )
        .collect()
    )


def build_coverage_df(*, restaurants_only: bool = True, low_thr: float = 3.0) -> pl.DataFrame:
    """
    Build a business-level table with coverage features:
      - photo_count
      - reviews_n
      - reviews with empty/non-empty text
      - flags: has_photos / has_reviews / has_text_reviews
      - stars_target: prefer review-mean stars, else fallback to business stars
    """
    business = build_universe(restaurants_only=restaurants_only)
    restaurant_ids_lf = business.select("business_id").lazy()

    photos_agg = aggregate_photos(restaurant_ids_lf)
    reviews_agg = aggregate_reviews(restaurant_ids_lf)

    df = (
        business.join(photos_agg, on="business_id", how="left")
        .join(reviews_agg, on="business_id", how="left")
        .with_columns(
            [
                pl.col("photo_count").fill_null(0).cast(pl.Int32),
                pl.col("reviews_n").fill_null(0).cast(pl.Int32),
                pl.col("reviews_text_empty_n").fill_null(0).cast(pl.Int32),
                pl.col("reviews_text_nonempty_n").fill_null(0).cast(pl.Int32),
                (pl.col("photo_count") > 0).alias("has_photos"),
                (pl.col("reviews_n") > 0).alias("has_reviews"),
                (pl.col("reviews_text_nonempty_n") > 0).alias("has_text_reviews"),
                (
                    pl.col("reviews_text_empty_n")
                    / pl.when(pl.col("reviews_n") == 0).then(None).otherwise(pl.col("reviews_n"))
                ).alias("share_reviews_empty_text"),
                pl.coalesce([pl.col("stars_review_mean"), pl.col("stars_business")]).alias("stars_target"),
            ]
        )
    )
    # low_thr is not used here; it is used by downstream summary/plots, but we keep the signature stable.
    _ = low_thr
    return df


def write_coverage_summary(out_path: Path, df: pl.DataFrame, *, restaurants_only: bool, low_thr: float) -> None:
    total = df.height

    def pct(x: int) -> float:
        return 100.0 * x / max(1, total)

    n_has_photos = df.filter(pl.col("has_photos")).height
    n_no_photos = total - n_has_photos

    n_has_reviews = df.filter(pl.col("has_reviews")).height
    n_no_reviews = total - n_has_reviews

    n_has_text = df.filter(pl.col("has_text_reviews")).height
    n_no_text = total - n_has_text

    n_reviews_no_photos = df.filter(pl.col("has_reviews") & (~pl.col("has_photos"))).height
    n_no_photos_no_text = df.filter((~pl.col("has_photos")) & (~pl.col("has_text_reviews"))).height

    n_low_all = df.filter(pl.col("stars_target") < low_thr).height
    n_low_with_photos = df.filter(pl.col("has_photos") & (pl.col("stars_target") < low_thr)).height
    n_low_no_photos = df.filter((~pl.col("has_photos")) & (pl.col("stars_target") < low_thr)).height

    out_path.write_text(
        "\n".join(
            [
                f"UNIVERSE: {'Restaurants only' if restaurants_only else 'All businesses'}",
                f"TOTAL BUSINESSES: {total}",
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def plot_basic_distributions(outdir: Path, df: pl.DataFrame) -> None:
    """
    Save a few simple plots to the output directory:
      - stars distribution (all vs with photos vs without photos)
      - stars vs photo_count
      - stars vs review_count
    """
    ensure_dir(outdir)

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


def merge_with_predictions(df: pl.DataFrame, pred_csv: Path) -> pl.DataFrame:
    """
    Join model predictions with the coverage table.
    """
    preds = pl.read_csv(pred_csv).with_columns(pl.col("business_id").cast(pl.Utf8))
    preds = preds.select(["business_id", "y_true", "y_pred", "error", "abs_error"])

    merged = preds.join(
        df.select(
            [
                "business_id",
                "photo_count",
                "reviews_n",
                "reviews_text_nonempty_n",
                "reviews_text_empty_n",
                "has_photos",
                "has_reviews",
                "has_text_reviews",
                "stars_target",
            ]
        ),
        on="business_id",
        how="left",
    )
    return merged


def plot_abs_error_vs_counts(outdir: Path, merged: pl.DataFrame) -> None:
    ensure_dir(outdir)

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


def mae_by_bucket(out_path: Path, merged: pl.DataFrame, *, col: str, edges: list[int], bucket_name: str) -> None:
    """
    Compute MAE grouped by a bucketized version of `col`, and save to CSV.
    """
    merged2 = merged.with_columns(
        [
            pl.col(col).fill_null(0).cast(pl.Int32).alias(f"{col}_i"),
            bucket_expr(pl.col(col).fill_null(0).cast(pl.Int32), edges, bucket_name),
        ]
    )
    out = (
        merged2.group_by(bucket_name)
        .agg(
            [
                pl.len().alias("n"),
                pl.col("abs_error").mean().alias("mae"),
            ]
        )
        .sort(bucket_name)
    )
    ensure_dir(out_path.parent)
    out.write_csv(out_path)
