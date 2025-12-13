"""
Dataframes and queries representing the raw dataset,
and a query with the features computed
"""

import polars as pl
from pathlib import Path

DATA_DIR = Path("data")

JSON_DIR = DATA_DIR / "dataset-json"
PATH_REVIEWS = JSON_DIR / "yelp_academic_dataset_review.json"
PATH_BUSINESS = JSON_DIR / "yelp_academic_dataset_business.json"

PHOTOS_DIR = DATA_DIR / "dataset-photos"


q_businesses = pl.scan_ndjson(PATH_BUSINESS)
q_reviews = pl.scan_ndjson(PATH_REVIEWS)


q_photos = pl.scan_ndjson(PHOTOS_DIR / "photos.json")

q_restaurants = q_businesses.filter(pl.col("categories").str.contains("Restaurants"))

# Queries

q_photos_agg = q_photos.group_by("business_id").agg(
    [
        pl.col("photo_id").alias("photo_ids"),
        pl.col("label").alias("photo_labels"),
        pl.len().alias("photo_count"),
    ]
)


q_exact_stars = (
    q_reviews.lazy()
    .select([pl.col("business_id"), pl.col("stars")])
    .group_by("business_id")
    .agg(pl.col("stars").mean().alias("exact_stars"))
)

q_features = (
    q_restaurants.join(q_photos_agg, on="business_id", how="inner")
    .join(q_exact_stars, on="business_id", how="inner")
    .select(
        # other interesting fields to select:
        # name, categories, attributes.RestaurantsPriceRange2, latitude/longitude, city, state, (etc)
        pl.col("business_id"),
        # pl.col("stars"),
        pl.col("exact_stars").alias("stars"),
        pl.col("review_count"),
        pl.col("photo_count"),
        pl.col("photo_ids"),
    )
    .with_columns(
        [pl.col("photo_ids").fill_null([]), pl.col("photo_count").fill_null(0)]
    )
)
