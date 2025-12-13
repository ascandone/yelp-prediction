import os
from pathlib import Path
import polars as pl

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path(os.environ.get("YELP_REPO_ROOT", "")) if os.environ.get("YELP_REPO_ROOT") else None,
        here.parents[2],            # .../src/yelp_prediction -> repo root
        Path.cwd(),
        Path.cwd().parent,
    ]
    for c in candidates:
        if c is None:
            continue
        if (c / "src" / "yelp_prediction").exists():
            return c
    return Path.cwd()

REPO_ROOT = _find_repo_root()
DATA_DIR = Path(os.environ.get("YELP_DATA_DIR", str(REPO_ROOT / "data")))

JSON_DIR = DATA_DIR / "dataset-json"
PATH_REVIEWS = JSON_DIR / "yelp_academic_dataset_review.json"
PATH_BUSINESS = JSON_DIR / "yelp_academic_dataset_business.json"

PHOTOS_DIR = DATA_DIR / "dataset-photos"
PATH_PHOTOS = PHOTOS_DIR / "photos.json"

def assert_data_available() -> None:
    required = [PATH_BUSINESS, PATH_REVIEWS, PATH_PHOTOS]
    missing = [p for p in required if not p.exists()]
    if missing:
        lines = "\n".join([f" - {p}" for p in missing])
        raise FileNotFoundError(
            "Missing Yelp dataset files:\n"
            f"{lines}\n\n"
            "Expected them under: <repo>/data/...\n"
            "Download the dataset (or set YELP_DATA_DIR to your data folder) and re-run."
        )

assert_data_available()

q_businesses = pl.scan_ndjson(PATH_BUSINESS)
q_reviews = pl.scan_ndjson(PATH_REVIEWS)
q_photos = pl.scan_ndjson(PATH_PHOTOS)
