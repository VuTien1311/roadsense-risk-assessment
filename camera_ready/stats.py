"""Small statistics helpers: Wilson intervals, image-level cluster bootstrap,
paired bootstrap differences and the exact McNemar test."""
from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_ROUNDS = 10000
CHUNK = 500  # bootstrap rounds per block, keeps memory bounded for 9,500-image splits


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes out of n trials."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, center - half, center + half


def cluster_bootstrap_ratio(numerators, denominators, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """Ratio of sums with a percentile CI from resampling images (clusters).

    ``numerators[i]`` and ``denominators[i]`` are the per-image counts, so
    objects from the same image always move together.
    """
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ratios = []
    for start in range(0, rounds, CHUNK):
        size = min(CHUNK, rounds - start)
        index = rng.integers(0, len(num), size=(size, len(num)))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios.append(num[index].sum(axis=1) / den[index].sum(axis=1))
    low, high = np.nanpercentile(np.concatenate(ratios), [2.5, 97.5])
    return float(num.sum() / den.sum()), float(low), float(high)


def paired_bootstrap_difference(values_a, values_b, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    """Mean(b - a) over images with a percentile CI (paired by image)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diff = b - a
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, rounds, CHUNK):
        size = min(CHUNK, rounds - start)
        index = rng.integers(0, len(diff), size=(size, len(diff)))
        means.append(diff[index].mean(axis=1))
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def mcnemar_exact(outcome_a, outcome_b):
    """Two-sided exact McNemar test for paired binary outcomes.

    Returns (b, c, p) where b counts a=1,b=0 and c counts a=0,b=1.
    """
    a = np.asarray(outcome_a, dtype=bool)
    b_arr = np.asarray(outcome_b, dtype=bool)
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    # Exact integers: 2.0**n overflows a float once n > 1023 (the train split has more discordant
    # images than that). int / int division is correctly rounded, so for n <= 1023 the p-value is
    # the same float as the former sum(comb) / 2.0**n.
    term = total = 1
    for i in range(k):
        term = term * (n - i) // (i + 1)  # comb(n, i + 1)
        total += term
    tail = total / (1 << n)
    return b, c, float(min(1.0, 2.0 * tail))


def fmt_ci(value, low, high, percent=True, digits=1):
    if value != value:  # NaN
        return "n/a"
    scale = 100.0 if percent else 1.0
    return f"{value * scale:.{digits}f} [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"
