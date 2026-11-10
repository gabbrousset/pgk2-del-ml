"""Writing the submission file from a score parquet

Validation split -> Team_NAME_UID.txt : 50 CatalogIDs, one per line.
Test split      -> Team_NAME_UID.csv : CatalogID,Score,Sel_50 for all test rows, with
                                       exactly 50 rows flagged Sel_50 == 1.

Column order is the template's, `CatalogID,Score,Sel_50`.
The wiki table lists the two flag columns the other way round.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.submit.selection import (
    butina_clusters,
    calibrate,
    greedy_expected_clusters,
    one_per_cluster,
    summarize,
)
from src.paths import SUBS, TEST_SPLIT

# ~431 kinase-assay-confirmed hits across the 428,960 released compounds,
# apportioned by split size
N_HITS = {"validation": 431 * 244328 / 428960, "test": 431 * 184632 / 428960}


def _read_selected(path: Path) -> list[str]:
    if path.suffix == ".txt":
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return pl.read_csv(path).filter(pl.col("Sel_50") == 1)["CatalogID"].to_list()


def pick(
    scores_path: Path,
    split: str,
    strategy: str,
    top_n: int,
    cutoff: float,
    exponent: float,
    exclude: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Returns all scored rows, the 50 picked rows, and a summary dict"""
    df = pl.read_parquet(scores_path).sort("score", descending=True)
    cand = df.head(top_n)

    print(f"clustering top {cand.height:,} candidates (Butina cutoff {cutoff})...")
    clusters = butina_clusters(cand["SMILES"].to_list(), cutoff=cutoff)
    print(f"  {len(set(clusters.tolist())):,} proxy clusters in candidate pool")

    if exclude is not None:
        # Whole clusters go, not just the picks.
        # The two submissions then cover different chemistry and get two shots at the 2-series bar
        prev = set(_read_selected(exclude))
        mask = np.array([c in prev for c in cand["CatalogID"].to_list()])
        dead = set(clusters[mask].tolist())
        keep = np.array([(c not in dead) for c in clusters.tolist()]) & ~mask
        print(f"  excluding {mask.sum()} prior picks and {len(dead)} of their clusters "
              f"-> {keep.sum():,} candidates remain")
        cand, clusters = cand.filter(pl.Series(keep)), clusters[keep]

    p = calibrate(cand["score"].to_numpy(), N_HITS[split], exponent=exponent)

    if strategy == "one_per_cluster":
        picks = one_per_cluster(p, clusters, k=50)
    else:
        picks, exp_clusters = greedy_expected_clusters(p, clusters, k=50)
        print(f"  greedy objective E[distinct clusters] = {exp_clusters:.3f}")

    info = summarize(picks, p, clusters, strategy)
    info.update(split=split, top_n=top_n, cutoff=cutoff, exponent=exponent,
                scores=scores_path.name)
    return df, cand[picks.tolist()], info


def write_validation(selected: pl.DataFrame, name: str) -> Path:
    ids = selected["CatalogID"].to_list()
    assert len(ids) == 50 and len(set(ids)) == 50, f"need 50 unique IDs, got {len(set(ids))}"
    path = SUBS / f"{name}.txt"
    path.write_text("\n".join(ids))
    print(f"wrote {path} ({len(ids)} IDs)")
    return path


def write_test(all_scores: pl.DataFrame, selected: pl.DataFrame, name: str) -> Path:
    ref = pl.read_csv(TEST_SPLIT)
    sel = set(selected["CatalogID"].to_list())
    assert len(sel) == 50, f"need exactly 50 selections, got {len(sel)}"

    out = (
        ref.select("CatalogID")
        .join(all_scores.select("CatalogID", "score"), on="CatalogID", how="left")
        .with_columns(
            pl.col("score").fill_null(0.0).alias("Score"),
            pl.col("CatalogID").is_in(list(sel)).cast(pl.Float64).alias("Sel_50"),
        )
        .select("CatalogID", "Score", "Sel_50")
    )
    assert out.height == ref.height == 184632, f"row count {out.height}"
    assert out["Sel_50"].sum() == 50, f"Sel_50 sums to {out['Sel_50'].sum()}"
    assert out["CatalogID"].n_unique() == out.height, "duplicate CatalogIDs"
    assert out["Score"].is_null().sum() == 0, "null scores"

    path = SUBS / f"{name}.csv"
    out.write_csv(path)
    print(f"wrote {path} ({out.height:,} rows, {int(out['Sel_50'].sum())} selected)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="score parquet, as written by src.model.score_test_config")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--name", required=True, help="output basename, e.g. Team_X_v1")
    ap.add_argument("--strategy", default="one_per_cluster",
                    choices=["one_per_cluster", "greedy"])
    ap.add_argument("--top-n", type=int, default=3000)
    ap.add_argument("--cutoff", type=float, default=0.65)
    ap.add_argument("--exponent", type=float, default=1.0)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="a previous submission file whose picks and clusters to avoid")
    args = ap.parse_args()

    SUBS.mkdir(parents=True, exist_ok=True)
    allsc, selected, info = pick(
        Path(args.scores), args.split, args.strategy, args.top_n, args.cutoff,
        args.exponent, args.exclude
    )
    if args.split == "validation":
        write_validation(selected, args.name)
    else:
        write_test(allsc, selected, args.name)
    print(info)


if __name__ == "__main__":
    main()
