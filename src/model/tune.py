"""Label recipes, and the training set built from one of them

The sweep behind `config/tune_results.json` needed our private assay results, so it is not here
"""

from __future__ import annotations

import polars as pl

from src.paths import DEDUP, ENUM

c = pl.col("count_PGK2"); ci = pl.col("count_PGK2_with_inhibitor")
ntc = pl.col("count_NTC"); hh = pl.col("historic_hits")
COMP = ci < 0.1 * c

# The wiki and the forum define a DEL hit differently.
# These are the six variants
RECIPES = {
    "wiki":     (c >= 3) & COMP & (ntc == 0) & (hh < 5),
    "strict":   (c > 3) & COMP & ((ntc == 0) | (ntc < 0.1 * c)) & (hh < 5),
    "allo":     (c >= 3) & ~COMP & (ntc == 0) & (hh < 5),
    "any":      (c >= 3) & (ntc == 0) & (hh < 5),
    "anyloose": (c >= 3) & (ntc == 0),
    "gated":    (c >= 5) & ((ntc == 0) | (ntc < 0.1 * c)) & (hh < 5)
                & pl.when(c >= 10).then(COMP).otherwise(pl.lit(True)),
}


def sample_enumeration(per_library: dict[str, int], seen: set[str]) -> pl.DataFrame:
    """Sample never-detected compounds, striding so the sample spans each parquet"""
    missing = [lib for lib, want in per_library.items() if want
               and not (ENUM / f"{lib}.enumeration.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"the enumerated library is needed to build training negatives, and "
            f"{len(missing)} of its files are absent: {', '.join(sorted(missing))}. "
            f"Looked under {ENUM}. See data/README.md. Training on a smaller "
            f"negative set would score differently while appearing to work.")
    parts = []
    for lib, want in per_library.items():
        if not want:
            continue
        path = ENUM / f"{lib}.enumeration.parquet"
        total = pl.scan_parquet(path).select(pl.len()).collect().item()
        stride = max(1, total // (want * 3))   # 3x headroom for dedup losses
        d = (
            pl.scan_parquet(path)
            .select("compound", "SMILES")
            .gather_every(stride)
            .collect()
            .filter(~pl.col("SMILES").is_in(list(seen)))
            .head(want)
            .with_columns(library=pl.lit(lib), count_PGK2=pl.lit(0, pl.Int64))
        )
        parts.append(d)
        print(f"  enum {lib:9s} total={total:>11,} sampled={d.height:>7,}")
    return pl.concat(parts)


def trainset(recipe: str, neg_ratio: int) -> pl.DataFrame:
    """Positives under `recipe`, plus `neg_ratio` enumeration negatives per positive

    Negatives are drawn per sub-library: the positive rate spans 133x across them, so an
    unstratified sample lets the model score "which library is this" instead.
    """
    df = pl.read_parquet(DEDUP)
    pos = df.filter(RECIPES[recipe])
    want = (pos.group_by("library").agg((pl.len() * neg_ratio).alias("n"))
               .to_dict(as_series=False))
    neg = sample_enumeration(dict(zip(want["library"], want["n"])),
                             set(df["SMILES"].to_list()))
    hard = df.filter((pl.col("sup_count_NTC") >= 1) & ~RECIPES[recipe])
    neg = pl.concat([neg.select("SMILES", "library", "count_PGK2"),
                     hard.select("SMILES", "library", "count_PGK2")])
    return pl.concat([
        pos.select("SMILES", "library", "count_PGK2").with_columns(y=pl.lit(1, pl.Int8)),
        neg.with_columns(y=pl.lit(0, pl.Int8)),
    ]).unique(subset="SMILES", keep="first")
