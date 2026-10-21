"""Step 16: Statistical analysis across repeated runs with different seeds."""
from typing import Dict, List

import numpy as np
from scipy import stats


def summarize_runs(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "values": arr.tolist(),
    }


def paired_t_test(a: List[float], b: List[float]) -> Dict:
    """Paired t-test between two conditions run over the *same* seeds
    (e.g. baseline accuracy vs. defended-model accuracy)."""
    t_stat, p_val = stats.ttest_rel(a, b)
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


def one_way_anova(*groups: List[float]) -> Dict:
    """One-way ANOVA across >2 conditions (e.g. multiple defenses)."""
    f_stat, p_val = stats.f_oneway(*groups)
    return {"f_stat": float(f_stat), "p_value": float(p_val)}


def format_mean_std(summary: Dict, decimals: int = 4) -> str:
    return f"{summary['mean']:.{decimals}f} \u00b1 {summary['std']:.{decimals}f}"


def cohens_d_paired(a: List[float], b: List[float]):
    """Paired-samples effect size d_z = mean(diff) / sd(diff). None if sd == 0."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 0 else None


def mean_diff_ci(a: List[float], b: List[float], conf: float = 0.95) -> Dict:
    """t-based confidence interval for the mean paired difference (a - b)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    m = float(diff.mean())
    if n < 2:
        return {"mean_diff": m, "ci_low": None, "ci_high": None, "conf": conf}
    se = float(diff.std(ddof=1) / np.sqrt(n))
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * se)
    return {"mean_diff": m, "ci_low": m - h, "ci_high": m + h, "conf": conf}


def paired_comparison(a: List[float], b: List[float]) -> Dict:
    """Paired t-test + Cohen's d_z + 95% CI of the mean difference."""
    return {**paired_t_test(a, b), "cohens_d": cohens_d_paired(a, b),
            **mean_diff_ci(a, b)}
