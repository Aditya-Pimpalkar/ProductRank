"""Statistical significance testing between two variants (PR-22).

A measured lift is only meaningful if it survives noise. Given per-query metric values
for two variants over the *same* queries, we run a **paired** test: paired because each
query is evaluated by both variants, so the natural comparison is per-query differences,
which removes query-difficulty as a confound and is far more powerful than an unpaired
test.

Two tests are offered:
  - Paired t-test (scipy) — parametric, the standard quick check.
  - Bootstrap CI on the mean paired difference — assumption-light, robust to the
    non-normal, bounded distributions IR metrics actually have.

Both are reported so a lift isn't claimed when it's within noise (ARCHITECTURE §3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    metric: str
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    t_statistic: float
    p_value: float
    ci_low: float  # bootstrap 95% CI on (b - a)
    ci_high: float
    n: int
    significant: bool  # p < alpha AND CI excludes 0

    def __str__(self) -> str:
        star = "✱" if self.significant else " "
        return (
            f"{self.metric}: {self.mean_a:.4f} → {self.mean_b:.4f} "
            f"(Δ={self.mean_diff:+.4f}, p={self.p_value:.4f}, "
            f"95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]) {star}"
        )


def paired_significance(
    a_per_query: dict[str, float],
    b_per_query: dict[str, float],
    metric: str = "ndcg_cut_10",
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> SignificanceResult:
    """Compare variant B against variant A on one metric across shared queries."""
    qids = sorted(set(a_per_query) & set(b_per_query))
    if len(qids) < 2:
        raise ValueError("Need at least 2 shared queries for a paired test.")

    a = np.array([a_per_query[q] for q in qids], dtype=float)
    b = np.array([b_per_query[q] for q in qids], dtype=float)
    diff = b - a

    # Paired t-test. Two degenerate (zero-variance) cases break the t-statistic:
    #   - all differences zero      → no effect at all          → p = 1.0
    #   - all differences equal ≠ 0 → a perfect, certain effect → p ≈ 0 (t → ∞)
    # Handle both explicitly; otherwise scipy returns nan with a divide-by-zero warning.
    if np.allclose(diff, 0):
        t_stat, p_value = 0.0, 1.0
    elif np.allclose(diff, diff[0]):
        t_stat, p_value = float("inf"), 0.0
    else:
        t_stat, p_value = stats.ttest_rel(b, a)

    # Bootstrap 95% CI on the mean paired difference.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    significant = bool(p_value < alpha and (ci_low > 0 or ci_high < 0))

    return SignificanceResult(
        metric=metric,
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(diff.mean()),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n=len(qids),
        significant=significant,
    )
