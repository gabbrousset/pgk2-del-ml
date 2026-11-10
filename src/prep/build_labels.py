"""Dedup and labelling of the DEL selection file

The dedup key and the aggregation are the organizers' data dictionary's
"""

import sys
from pathlib import Path

import polars as pl
from src.model.tune import RECIPES
from src.paths import RAW, NTC, INTERIM as OUT


def dedup(src: Path) -> pl.DataFrame:
    lf = pl.scan_parquet(src)
    n_raw = lf.select(pl.len()).collect().item()

    # carbon test must accept lowercase aromatic carbon
    has_carbon = pl.col("SMILES").str.count_matches("c|C") > 0
    plausible_length = pl.col("SMILES").str.len_chars() > 10
    well_formed_id = pl.col("compound").str.contains(r"^qDOS[0-9_]+(-[0-9]+){2,3}$")
    lf = lf.filter(well_formed_id & has_carbon & plausible_length)
    n_clean = lf.select(pl.len()).collect().item()

    zcols = ["zscore_PGK2", "zscore_PGK2_with_inhibitor", "zscore_NTC"]
    agg = (
        [pl.col("compound").first()]
        + [pl.col(c).sum() for c in ("count_PGK2", "count_PGK2_with_inhibitor", "count_NTC")]
        + [(pl.col(c).sum() / pl.len().sqrt()).alias(c) for c in zcols]   # Stouffer
        + [pl.col("historic_hits").max(), pl.len().alias("n_encodings")]
    )
    df = lf.group_by("SMILES").agg(agg).collect()

    df = df.with_columns(
        pl.col("compound").str.extract(r"^(qDOS[0-9_]+)-").alias("library"),
        pl.col("compound").str.extract(r"-([0-9]+)-[0-9]+-[0-9]+$").cast(pl.Int32).alias("bb1"),
        pl.col("compound").str.extract(r"-([0-9]+)-[0-9]+$").cast(pl.Int32).alias("bb2"),
        pl.col("compound").str.extract(r"-([0-9]+)$").cast(pl.Int32).alias("bb3"),
    )
    print(f"raw={n_raw:,} cleaned={n_clean:,} unique_smiles={df.height:,}", file=sys.stderr)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = dedup(RAW)

    # structures seen only in the no-target arm: matrix binders, used as hard negatives
    sup = (
        pl.scan_parquet(NTC)
        .select("SMILES", pl.col("count_NTC").alias("sup_count_NTC"))
        .group_by("SMILES")
        .agg(pl.col("sup_count_NTC").sum())
        .collect()
    )
    df = df.join(sup, on="SMILES", how="left").with_columns(
        pl.col("sup_count_NTC").fill_null(0)
    )

    df.write_parquet(OUT / "selection_dedup.parquet", compression="zstd")
    print(f"\nwrote {OUT / 'selection_dedup.parquet'} ({df.height:,} rows)")

    for name, rule in RECIPES.items():
        print(f"  {name:10s} {df.filter(rule).height:>8,} positives")

    print("\npositives per library (anyloose):")
    print(
        df.with_columns(RECIPES["anyloose"].alias("pos"))
        .group_by("library")
        .agg(pl.len().alias("n"), pl.col("pos").sum().alias("pos"))
        .with_columns((pl.col("pos") / pl.col("n") * 100).round(4).alias("pos_pct"))
        .sort("pos", descending=True)
    )


if __name__ == "__main__":
    main()
