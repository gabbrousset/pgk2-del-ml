"""Fingerprints, cached on disk

uint8 rather than bits: LightGBM wants dense input anyway
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.paths import FP as CACHE
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg

RDLogger.DisableLog("rdApp.*")


NBITS = 2048

# `ecfp4c` is the count fingerprint (not the binary one)
_COUNT = {"ecfp4c"}

# The four fingerprints named in config/tune_results.json
_GENS = {
    "ecfp4": lambda: rfg.GetMorganGenerator(radius=2, fpSize=NBITS),
    "ecfp4c": lambda: rfg.GetMorganGenerator(radius=2, fpSize=NBITS),
    "fcfp6": lambda: rfg.GetMorganGenerator(
        radius=3, fpSize=NBITS, atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen()
    ),
    "atompair": lambda: rfg.GetAtomPairGenerator(fpSize=NBITS),
}

_WORKER: dict = {}


def _init(name: str) -> None:
    _WORKER["gen"] = _GENS[name]()
    _WORKER["count"] = name in _COUNT


def _run(smiles_chunk: list[str]) -> np.ndarray:
    gen, count = _WORKER["gen"], _WORKER["count"]
    take = gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy
    out = np.zeros((len(smiles_chunk), NBITS), dtype=np.uint8)
    for i, smi in enumerate(smiles_chunk):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        out[i] = np.minimum(take(mol), 255)  # clips at 255, max seen ~10
    return out


def _warn_dead(arr: np.ndarray, name: str, cached: bool) -> int:
    """Warn if rows are all-zero: those inputs did not parse

    Runs on the cache path too, because a cached bad matrix comes back silently.
    """
    if len(arr) == 0:
        return 0
    dead = int((~arr.any(axis=1)).sum())
    if not dead:
        return 0
    pct = 100 * dead / len(arr)
    where = "loaded from cache" if cached else "about to be cached"
    print(f"  !! {name}: {dead:,} of {len(arr):,} rows ({pct:.2f}%) have NO BITS "
          f"SET, which is almost always a parse failure. If you passed IDs instead of "
          f"SMILES this matrix is meaningless, and it is {where} under that input's hash.")
    if pct > 1:
        print(f"  !! {name}: above 1%, so treat every number computed from this "
              f"matrix as void until you have checked the input.")
    return dead


def _key(name: str, smiles: list[str]) -> Path:
    h = hashlib.sha1(("\n".join(smiles)).encode()).hexdigest()[:16]
    return CACHE / f"{name}_{len(smiles)}_{h}.npy"


def featurize(
    smiles: list[str], name: str = "ecfp4", n_jobs: int | None = None, chunk: int = 2000
) -> np.ndarray:
    """Return an (n, 2048) uint8 fingerprint matrix, cached on disk"""
    if name not in _GENS:
        raise KeyError(f"unknown fingerprint {name!r}; have {sorted(_GENS)}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _key(name, smiles)
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        _warn_dead(arr, name, cached=True)
        return arr

    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 1)
    chunks = [smiles[i : i + chunk] for i in range(0, len(smiles), chunk)]
    parts: list[np.ndarray] = []
    with ProcessPoolExecutor(n_jobs, initializer=_init, initargs=(name,)) as ex:
        for j, part in enumerate(ex.map(_run, chunks)):
            parts.append(part)
            if j % 50 == 0:
                done = min((j + 1) * chunk, len(smiles))
                print(f"  {name}: {done:,}/{len(smiles):,}", end="\r", flush=True)
    arr = np.vstack(parts)
    print(f"  {name}: {len(smiles):,} done" + " " * 20)

    _warn_dead(arr, name, cached=False)

    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)  # atomic, so a killed run never leaves a half-written cache
    return arr


def stack(smiles: list[str], names: list[str], **kw) -> np.ndarray:
    """Horizontally concatenate several fingerprint families"""
    return np.hstack([np.asarray(featurize(smiles, n, **kw)) for n in names])
