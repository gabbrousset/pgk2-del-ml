"""Rebuild and scoring of one tuned configuration

Use `--from-model` on a new library: a retrain reproduces these scores today, and
stops doing so once LightGBM or NumPy moves.
Retrained boosters go to `logs/`; `models/` is never written to.

    python -m src.model.score_test_config --tag tune20 --from-model models/tune20t.txt
    python -m src.model.score_test_config --tag tune20
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from src.prep.features import stack
from src.model.tune import trainset
from src.paths import HERE, LOGS, MODELS, PROC, RESULTS, TUNE, VT


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--from-model", default=None, metavar="PATH",
                    help="score from a saved booster. The fingerprint stack still comes "
                         "from the config, so the tag must match the saved model.")
    ap.add_argument("--recipe", default=None,
                    help="override the config's positive-label recipe. The README's "
                         "label ablation was made this way.")
    ap.add_argument("--splits", nargs="+", default=["Test"], choices=["Test"],
                    help="which splits to score")
    args = ap.parse_args()
    cfg = dict({x["tag"]: x for x in json.loads(TUNE.read_text())}[args.tag])
    if args.recipe:
        cfg["recipe"] = args.recipe
    out = args.out_tag or f"{args.tag}t"
    print({k: cfg[k] for k in ("recipe", "fps", "neg_ratio", "num_leaves", "learning_rate",
                               "n_estimators", "min_child_samples", "feature_fraction",
                               "lambda_l2", "scale_pos_weight")})
    fps = cfg["fps"].split(",")
    prov = {"tag": args.tag, "out_tag": out, "config": cfg, "fps": fps}

    if args.from_model:
        booster = lgb.Booster(model_file=args.from_model)
        prov["scored_from"] = str(args.from_model)
        prov["booster_sha256"] = _sha(Path(args.from_model))
        print(f"scoring from saved booster {args.from_model} "
              f"({booster.num_trees()} trees, sha256 {prov['booster_sha256'][:12]})")
        predict = lambda M: booster.predict(M)
    else:
        tr = trainset(cfg["recipe"], cfg["neg_ratio"])
        X = stack(tr["SMILES"].to_list(), fps); y = tr["y"].to_numpy()
        spw = cfg["scale_pos_weight"] or float((y == 0).sum() / max(1, (y == 1).sum()))
        print(f"pos {int(y.sum()):,}  neg {int((1 - y).sum()):,}  scale_pos_weight {spw:.3f}")
        m = lgb.LGBMClassifier(objective="binary", n_jobs=-1, verbosity=-1,
            num_leaves=cfg["num_leaves"], learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"], min_child_samples=cfg["min_child_samples"],
            feature_fraction=cfg["feature_fraction"], lambda_l2=cfg["lambda_l2"],
            scale_pos_weight=spw).fit(X, y)
        # `models/` holds the booster this repo ships and is never written to here
        mp = LOGS / f"{out}.txt"
        mp.parent.mkdir(parents=True, exist_ok=True)
        m.booster_.save_model(str(mp))
        prov.update(train_rows=int(len(y)), positives=int(y.sum()),
                    scale_pos_weight=spw, booster=str(mp.relative_to(HERE)),
                    booster_sha256=_sha(mp), n_trees=int(m.booster_.num_trees()))
        print(f"saved booster -> {mp.relative_to(HERE)} "
              f"({m.booster_.num_trees()} trees, sha256 {prov['booster_sha256'][:12]})")
        predict = lambda M: m.predict_proba(M)[:, 1]

    for d in (PROC, RESULTS, MODELS):
        d.mkdir(parents=True, exist_ok=True)

    prov["scores"] = {}
    for split in args.splits:
        d = pl.read_csv(VT / f"PGK2_{split}_split.csv")
        p = predict(stack(d["SMILES"].to_list(), fps))
        f = PROC / f"scores_{split.lower()}_{out}.parquet"
        d.with_columns(score=pl.Series(p)).sort("score", descending=True).write_parquet(f)
        prov["scores"][split.lower()] = {"file": f.name, "sha256": _sha(f),
                                         "rows": int(d.height)}
        print(f"  {split}: top {p.max():.4f} median {np.median(p):.5f} -> {f.name}")

    (RESULTS / f"provenance_{out}.json").write_text(json.dumps(prov, indent=1))
    print(f"  provenance -> logs/results/provenance_{out}.json")


if __name__ == "__main__":
    main()
