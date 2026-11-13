"""mcnemar_exact after the overflow fix: identical floats to the old formula for n <= 1023,
correct (vs scipy) for the large n of the train split."""
import math
import random
import sys

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from stats import mcnemar_exact  # noqa: E402


def old(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * (sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)))


def outcomes(b, c, same=7):
    a = [1] * b + [0] * c + [1] * same + [0] * same
    d = [0] * b + [1] * c + [1] * same + [0] * same
    return np.array(a), np.array(d)


checked = 0
for n in range(0, 301):
    for b in range(0, n + 1):
        got = mcnemar_exact(*outcomes(b, n - b))
        assert got[:2] == (b, n - b), got
        assert got[2] == old(b, n - b), (b, n - b, got[2], old(b, n - b))
        checked += 1
rng = random.Random(1)
for _ in range(3000):
    n = rng.randint(301, 1023)
    b = rng.randint(0, n)
    assert mcnemar_exact(*outcomes(b, n - b))[2] == old(b, n - b), (b, n - b)
    checked += 1
print("identical to the old formula for", checked, "cases with n <= 1023")

worst = 0.0
for b, c in [(61, 94), (500, 600), (900, 1100), (2000, 2100), (30394, 35144), (35000, 35001), (1, 5000),
             (0, 69863), (12000, 13000), (4000, 3890)]:
    p = mcnemar_exact(*outcomes(b, c))[2]
    ref = float(min(1.0, 2.0 * binom.cdf(min(b, c), b + c, 0.5)))
    rel = abs(p - ref) / ref if ref > 0 else abs(p)
    worst = max(worst, rel if ref > 1e-300 else 0.0)
    print(f"b={b} c={c} p={p:.6g} scipy={ref:.6g}")
    assert (ref < 1e-300 and p < 1e-300) or rel < 1e-9, (b, c, p, ref)
print("matches scipy for large n, worst relative difference", worst)
