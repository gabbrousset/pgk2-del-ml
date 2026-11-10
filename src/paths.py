"""Every path this repo uses"""
from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "src").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("repo root not found above " + str(Path(__file__).resolve()))


HERE = _root()

# From the Synapse download
DATA       = Path(os.environ.get("PGK2_DATA") or HERE / "data")
DEL_DIR    = DATA / "Target2035 PGK2 DEL-ML Challenge"
RAW        = DEL_DIR / "PGK2_selection.parquet"
NTC        = DEL_DIR / "PGK2_NTC_supplement.parquet"
ENUM       = DEL_DIR / "OpeDELLibrary"        # Training negatives are sampled from here
VT         = DEL_DIR / "Val-Test-set"
TEST_SPLIT = VT / "PGK2_Test_split.csv"

# Written by the pipeline
INTERIM = DATA / "interim"
PROC    = DATA / "processed"
FP      = INTERIM / "fp"
DEDUP   = INTERIM / "selection_dedup.parquet"

# Under the repo root
MODELS  = HERE / "models"
CONFIG  = HERE / "config"
TUNE    = CONFIG / "tune_results.json"
LOGS    = HERE / "logs"
RESULTS = LOGS / "results"
SUBS    = HERE / "submissions"

REQUIRED_TO_SCORE   = ("TEST_SPLIT",)
REQUIRED_TO_RETRAIN = ("RAW", "NTC", "ENUM")
DERIVED             = ("INTERIM", "PROC", "FP", "DEDUP", "LOGS", "RESULTS", "SUBS")


def check() -> int:
    """Print which inputs are present, and whether a retrain or only scoring will run

        python -m src.paths
    """
    g = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, Path)}
    groups = (("needed to score from the saved booster", REQUIRED_TO_SCORE),
              ("needed as well to retrain", REQUIRED_TO_RETRAIN),
              ("created by the pipeline", DERIVED))
    named = {k for _, ks in groups for k in ks}
    for title, keys in groups:
        print(f"\n  {title}:")
        for k in keys:
            print(f"    {'ok     ' if g[k].exists() else 'MISSING'}  {k:<12} {g[k]}")
    print("\n  other paths:")
    for k in sorted(set(g) - named):
        print(f"    {'ok     ' if g[k].exists() else 'MISSING'}  {k:<12} {g[k]}")

    missing_score = [k for k in REQUIRED_TO_SCORE if not g[k].exists()]
    missing_train = [k for k in REQUIRED_TO_RETRAIN if not g[k].exists()]
    print()
    if missing_score:
        print(f"cannot score: {', '.join(missing_score)} missing. See data/README.md.")
        return 1
    if missing_train:
        print(f"can score from the saved booster, but cannot retrain: "
              f"{', '.join(missing_train)} missing. See data/README.md.")
        return 0
    print("all inputs present: both scoring and a full retrain will run")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
