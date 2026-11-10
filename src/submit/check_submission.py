"""Format checks for a blind-test submission, to run before sending it

    Usecols do not match columns, columns expected but not found:
        ['CatalogID', 'Score', 'Sel_50']        -- a .txt file, or a wrong header
    Missing molecules, expected 178052 unique molecules, found 184632
                                                -- the wrong split
    Missing molecules, expected 178052 unique molecules, found 178049
                                                -- rows silently dropped
    Molecules not recognized: <CatalogID>,...   -- compounds outside the split

    python -m src.submit.check_submission submissions/*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
from src.paths import TEST_SPLIT

HEADER = "CatalogID,Score,Sel_50"


def check(path: Path, want: set[str]) -> bool:
    if path.suffix != ".csv":
        print(f"\n{path.name}  *** REJECT ***\n    FAIL  the file must be .csv "
              f"(the board rejects .txt with 'Usecols do not match columns')")
        return False

    header = path.read_text().split("\n", 1)[0].strip()
    d = pl.read_csv(path)
    # Every check guards on the column it reads. A file missing `Sel_50` is the board's
    # first error string, so this has to report it rather than raise on it.
    has = d.columns.__contains__
    ids = set(d["CatalogID"].to_list()) if has("CatalogID") else set()
    sel = (d.filter(pl.col("Sel_50") == 1)["CatalogID"].to_list()
           if has("Sel_50") and has("CatalogID") else [])
    score = d["Score"] if has("Score") else None

    checks = {
        f"header is exactly {HEADER!r}": header == HEADER,
        f"row count is {len(want):,}": d.height == len(want),
        "CatalogIDs unique": has("CatalogID") and d["CatalogID"].n_unique() == d.height,
        "CatalogIDs match the split exactly": ids == want,
        "Sel_50 sums to exactly 50": has("Sel_50") and int(d["Sel_50"].sum()) == 50,
        "50 distinct selected molecules": len(set(sel)) == 50,
        "Sel_50 is only 0 or 1": has("Sel_50")
            and set(d["Sel_50"].unique().to_list()) <= {0.0, 1.0, 0, 1},
        "no null scores": score is not None and score.is_null().sum() == 0,
        "no NaN scores": score is not None
            and (not score.dtype.is_float() or score.is_nan().sum() == 0),
        "Score varies (AUC computable)": score is not None and score.n_unique() > 1000,
    }

    ok = all(checks.values())
    print(f"\n{path.name}  {'OK' if ok else '*** REJECT ***'}")
    for k, v in checks.items():
        if not v:
            print(f"    FAIL  {k}")
    if ids != want:
        print(f"          {len(ids - want):,} CatalogIDs not in the split, "
              f"{len(want - ids):,} split CatalogIDs absent")
    if ok:
        print(f"    all {len(checks)} checks pass; {len(set(sel))} molecules selected")
    return ok


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    want = set(pl.read_csv(TEST_SPLIT)["CatalogID"].to_list())
    print(f"blind-test split: {len(want):,} CatalogIDs")
    allok = all([check(Path(p), want) for p in sys.argv[1:]])
    print(f"\n{'ALL FILES VALID' if allok else 'SOME FILES INVALID: DO NOT SUBMIT'}")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
