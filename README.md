# pgk2-del-ml

Machine learning on DNA-encoded library screening data, to find small-molecule binders of phosphoglycerate kinase 2 (PGK2).

Submitted to the blind test step of the DREAM × CACHE Target 2035 challenge as Team McGillians, where it returned 4 chemical series from 5 confirmed hits.
Ranking it 2nd on the submissions leaderboard.

**Team McGillians:**
- Gabriel Caballero
- Adrián Hernández Nájera

## goal

Challenge rules, data and metric: https://www.synapse.org/Synapse:syn75349604/wiki/

Binders in as many distinct chemical series as possible, rather than as many binders as possible.

## method

A gradient-boosted classifier (LightGBM, 150 trees, 31 leaves) over count-based ECFP4 fingerprints, used to rank a 184,632-compound catalogue split.

It trains on 324,698 compounds: 28,863 positives taken from the DEL selection counts, and 295,835 negatives.
Most negatives are ten per positive from the enumerated library, the full list of what the building blocks and reactions could produce.
The rest are structures the screen saw only in the no-target control.

A selection layer then clusters the top 3,000 with Butina at a distance cutoff of 0.65 (Tanimoto similarity 0.35) and takes the best-scoring member of each cluster.
So the 50 picks span 50 distinct clusters.

The selection also takes another submission as input and drops its picks, along with every cluster they sit in, so two files cover different chemistry rather than overlapping.
Ours was passed `test_A_sim65`, the other file we sent that round.

## label definition

Model, features, hyperparameters and selection rule held fixed, varying only the positive-label rule.
Scored by where the 39 later-revealed actives land in the ranked test split:

| label rule | top 50 | top 500 | ROC AUC (test) |
|---|---|---|----------------|
| `wiki`, the organizers' published recipe | 1 | 4 | 0.8795         |
| `strict`, the forum recommendation | 0 | 1 | 0.8475         |
| `gated`, five reads minimum, inhibitor test from ten reads up | 2 | 3 | 0.8425         |
| `allo`, signal that survives the inhibitor | 0 | 4 | 0.8118         |
| `any`, `wiki` without the inhibitor test | 1 | 6 | 0.8769         |
| `anyloose`, what was submitted | 2 | 9 | 0.8895         |

`anyloose` is `count_PGK2 >= 3 AND count_NTC == 0`: at least three reads on target, none in the no-target control.
It drops both filters `wiki` applies: whether signal falls when a known inhibitor is present, and whether the compound has hit many past screens.
All six are in `src/model/tune.py`.

Note: The table is retrospective, `anyloose` was picked on 2 October against validation actives we had confirmed with our own assay credits.
The 39 test actives were revealed later, and this comparison was computed then.


## other approaches

We tried other methods but none ranked the 39 actives better than the model above.
Some we did not have time to explore properly.

- docking (FITTED)
- 3D shape (USR, USRCAT)
- graph neural networks (ChemProp)
- chemical language models (MoLFormer-XL, ChemBERTa-2)
- synthon / disynthon enrichment

## running it

`data/README.md` says where to get the data.
`uv run python -m src.paths` reports what you have.

```bash
uv sync

uv run python -m src.model.score_test_config --tag tune20 \
    --from-model models/tune20t.txt

uv run python -m src.submit.make_submission \
    --scores data/processed/scores_test_tune20t.parquet \
    --split test --cutoff 0.65 --top-n 3000 \
    --exclude Team_McGillians_test_A_sim65.csv \
    --name Team_McGillians_test_D_delml

uv run python -m src.submit.check_submission submissions/Team_McGillians_test_D_delml.csv
```

`Team_McGillians_test_A_sim65.csv` is not in this repo, so without `--exclude` the commands rebuild 48 of the 50.
Dropping `--from-model` retrains, which also needs the enumerated library.

`config/tune_results.json` holds all 26 swept configurations.
The sweep only nominated candidates, and `tune20` did the best of the five we tested on validation, at p = 0.0013.

On an M1 Pro the full retrain takes about 100 seconds: 8 s to deduplicate the 7.7M-row selection file, 91 s to build the training set, fingerprint 509,330 molecules and fit, 2 s to select the 50.
