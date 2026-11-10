# data

Not in this repo.
The organizers distribute it: https://www.synapse.org/Synapse:syn75349604/wiki/641046

## files

Scoring from the saved booster needs one file, 10 MB.
Retraining needs all four, about 5.6 GB.

| file | size | needed for |
|---|---|---|
| `Val-Test-set/PGK2_Test_split.csv` | 10 MB | both |
| `PGK2_selection.parquet` | 193 MB | retrain |
| `PGK2_NTC_supplement.parquet` | 10 MB | retrain |
| `OpeDELLibrary/*.enumeration.parquet` | 5.4 GB, 12 files | retrain |

Paths are relative to `Target2035 PGK2 DEL-ML Challenge/` in the download.

Most training negatives come from `OpeDELLibrary/`.
The rest are structures the selection file saw only in the no-target arm.
The code reads only the libraries holding positives under the chosen recipe, 10 of the 12 for `anyloose`.
Which 10 depends on the recipe, so fetch the whole directory.
A missing file is an error rather than a warning, because training on a smaller negative set would score differently while appearing to work.

The submission was built from these files, checksummed 2026-11-09.
The enumeration is 12 files and is not listed.

    be2de34a65e1efbe9fe17b228d8713806855dee547af1e8e3896e9169b33795f  PGK2_Test_split.csv
    1dab30c45a91e83e48886c4a0def0a873eefd8d3211d886e865e89369ad61a13  PGK2_selection.parquet
    bb9868a174b69d4d9cabcad61e40a0585777c02a24c6dc767bfb222e40d2b4d0  PGK2_NTC_supplement.parquet

    shasum -a 256 "data/Target2035 PGK2 DEL-ML Challenge/PGK2_selection.parquet"

## where to put it

Unpack it unchanged:

    data/
      Target2035 PGK2 DEL-ML Challenge/
        PGK2_selection.parquet
        PGK2_NTC_supplement.parquet
        OpeDELLibrary/
        Val-Test-set/
          PGK2_Test_split.csv

To keep it elsewhere, point the code at it:

    export PGK2_DATA=/path/to/your/download

`uv run python -m src.paths` prints every path the code uses and says which are missing.
`data/interim/` and `data/processed/` are created by the pipeline and gitignored.
A retrain writes about 1 GB of intermediate parquet and 1.2 GB of cached fingerprints.
