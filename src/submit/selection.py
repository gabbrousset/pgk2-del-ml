"""Clustering and the pick rules"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")


def calibrate(scores: np.ndarray, expected_hits: float, exponent: float = 1.0) -> np.ndarray:
    """Map raw scores to probabilities summing to `expected_hits`

    Greedy needs p on an absolute scale, not just ranks: at p ~ 1 it says spread out,
    at p ~ 0 it says chase score.
    `exponent` > 1 raises p at the top, which makes greedy discount a repeat harder,
    so a higher exponent spreads picks more rather than less.
    """
    s = np.clip(scores.astype(np.float64), 1e-12, None) ** exponent
    p = s * (expected_hits / s.sum())
    return np.clip(p, 0.0, 0.95)


def butina_clusters(smiles: list[str], cutoff: float = 0.65, radius: int = 2) -> np.ndarray:
    """Cluster by ECFP Tanimoto. cutoff is a DISTANCE (1 - Tanimoto)

    The organizers' proxy is agglomerative average linkage at the same similarity.
    We use Butina because it is cheaper, and did not compare the two
    """
    gen = rfg.GetMorganGenerator(radius=radius, fpSize=2048)
    fps, keep = [], []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            fps.append(gen.GetFingerprint(m))
            keep.append(i)

    # Condensed lower triangle, preallocated.
    # At `--top-n 10000` a Python list is 50M boxed floats and becomes the bottleneck
    n = len(fps)
    dists = np.empty(n * (n - 1) // 2, dtype=np.float64)
    pos = 0
    for i in range(1, n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
        dists[pos : pos + i] = 1.0 - sims
        pos += i

    labels = np.full(len(smiles), -1, dtype=np.int64)
    for cid, members in enumerate(Butina.ClusterData(dists, n, cutoff, isDistData=True)):
        for m in members:
            labels[keep[m]] = cid
    nxt = labels.max() + 1
    for i in np.where(labels < 0)[0]:   # unparsable molecules become singletons
        labels[i] = nxt
        nxt += 1
    return labels


def greedy_expected_clusters(
    p: np.ndarray, clusters: np.ndarray, k: int = 50, rho: float = 0.7
) -> tuple[np.ndarray, float]:
    """Greedily maximise expected distinct clusters hit, returning (indices, F)

    With per-compound hit probability p_i over proxy series c, the expected number of
    distinct series a pick set S hits is

        F(S) = sum_c [ 1 - prod_{i in S and c} (1 - p_i) ]

    F is monotone submodular, so greedy is within (1 - 1/e) of optimal at rho = 0.
    At rho > 0 the k-th repeat in a series is damped by (1 - rho)**k, which is no longer
    F's marginal gain, so that guarantee does not carry.
    The damping is deliberate: F alone discounts a repeat by `p_i`, 5-15% here, which
    collapses greedy to plain top-50.
    A series of analogues is not independent either.
    rho = 1 is strict one-per-cluster, 0.7 leans toward spreading.
    """
    miss: dict[int, float] = {}
    taken: dict[int, int] = {}
    chosen: list[int] = []
    available = np.ones(len(p), dtype=bool)

    for _ in range(min(k, len(p))):
        m = np.array([miss.get(c, 1.0) for c in clusters])
        disc = np.array([(1.0 - rho) ** taken.get(c, 0) for c in clusters])
        gain = np.where(available, m * p * disc, -np.inf)
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            break
        chosen.append(i)
        available[i] = False
        c = int(clusters[i])
        miss[c] = miss.get(c, 1.0) * (1.0 - p[i])
        taken[c] = taken.get(c, 0) + 1

    total = sum(1.0 - v for v in miss.values())
    return np.array(chosen, dtype=int), float(total)


def one_per_cluster(p: np.ndarray, clusters: np.ndarray, k: int = 50) -> np.ndarray:
    """One pick per cluster: its best-scoring member, clusters in score order"""
    order = np.argsort(-p)
    seen: set[int] = set()
    picks: list[int] = []
    for i in order:
        c = int(clusters[i])
        if c not in seen:
            seen.add(c)
            picks.append(int(i))
            if len(picks) == k:
                return np.array(picks, dtype=int)

    chosen = set(picks)
    for i in order:   # relax one-per-cluster rather than return fewer than k
        if int(i) not in chosen:
            picks.append(int(i))
            chosen.add(int(i))
            if len(picks) == k:
                break

    out = np.array(picks[:k], dtype=int)
    assert len(set(out.tolist())) == len(out), "one_per_cluster produced duplicates"
    return out


def summarize(picks: np.ndarray, p: np.ndarray, clusters: np.ndarray, label: str) -> dict:
    cl = clusters[picks]
    info = dict(
        strategy=label,
        n=len(picks),
        distinct_clusters=int(len(set(cl.tolist()))),
        sum_p=float(p[picks].sum()),
        max_p=float(p[picks].max()),
        min_p=float(p[picks].min()),
    )
    print(
        f"{label:28s} picks={info['n']:>3} distinct_proxy_clusters="
        f"{info['distinct_clusters']:>3} E[hits]={info['sum_p']:.2f}"
    )
    return info
