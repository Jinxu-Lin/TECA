# Component: Statistical Testing
# Source: research/method-design.md §4
# Ablation config key: N/A (always on)

"""Statistical analysis: paired t-tests, Cohen's d, and summary reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a paired statistical comparison."""

    test_name: str
    mean_real: float
    mean_null: float
    std_real: float
    std_null: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_low: float   # 95% CI for Cohen's d
    ci_high: float
    n: int
    passed: bool  # whether pass criteria are met


def paired_t_test(
    real_values: List[float],
    null_values: List[float],
    alpha: float = 0.05,
    min_cohens_d: float = 0.5,
    test_name: str = "TECS(real) vs TECS(null)",
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
) -> TestResult:
    """Run a paired t-test with Cohen's d effect size.

    Args:
        real_values: TECS values under the real condition.
        null_values: TECS values under the null condition.
        alpha: Significance level.
        min_cohens_d: Minimum Cohen's d for practical significance.
        bootstrap_n: If > 0, use bootstrap CI instead of normal approximation.
        bootstrap_seed: Seed for bootstrap resampling.

    Returns:
        TestResult with all statistics and pass/fail decision.
    """
    real = np.array(real_values)
    null = np.array(null_values)
    diff = real - null
    n = len(diff)

    t_stat, p_val = stats.ttest_rel(real, null)

    # Cohen's d for paired samples
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # CI for Cohen's d
    if bootstrap_n > 0:
        # Bootstrap CI
        rng = np.random.RandomState(bootstrap_seed)
        d_boot = []
        for _ in range(bootstrap_n):
            idx = rng.randint(0, n, size=n)
            diff_b = diff[idx]
            std_b = np.std(diff_b, ddof=1)
            if std_b > 0:
                d_boot.append(np.mean(diff_b) / std_b)
        if d_boot:
            ci_low = float(np.percentile(d_boot, 2.5))
            ci_high = float(np.percentile(d_boot, 97.5))
        else:
            ci_low, ci_high = float(d), float(d)
    else:
        # Normal approximation (non-central t)
        se_d = np.sqrt(1 / n + d ** 2 / (2 * n))
        ci_low = d - 1.96 * se_d
        ci_high = d + 1.96 * se_d

    passed = bool((p_val < alpha) and (abs(d) >= min_cohens_d))

    return TestResult(
        test_name=test_name,
        mean_real=float(np.mean(real)),
        mean_null=float(np.mean(null)),
        std_real=float(np.std(real, ddof=1)),
        std_null=float(np.std(null, ddof=1)),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        cohens_d=float(d),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n=n,
        passed=passed,
    )


def check_pass_criteria(
    test_real_vs_null: TestResult,
    test_edit_vs_placebo: TestResult,
    mean_tecs: float,
    min_tecs_mean: float = 0.05,
    angular_variance: float = 1.0,
    angular_kill_threshold: float = 0.001,
) -> dict:
    """Evaluate all probe pass criteria.

    Returns:
        Dict with per-criterion results and overall pass/fail.
    """
    results = {
        "criterion_1_statistical_significance": {
            "passed": test_real_vs_null.p_value < 0.05,
            "p_value": test_real_vs_null.p_value,
        },
        "criterion_2_effect_size": {
            "passed": abs(test_real_vs_null.cohens_d) >= 0.5,
            "cohens_d": test_real_vs_null.cohens_d,
        },
        "criterion_3_tecs_mean": {
            "passed": mean_tecs > min_tecs_mean,
            "mean_tecs": mean_tecs,
        },
        "criterion_4_placebo": {
            "passed": test_edit_vs_placebo.passed,
            "p_value": test_edit_vs_placebo.p_value,
            "cohens_d": test_edit_vs_placebo.cohens_d,
        },
        "kill_gate_angular_variance": {
            "killed": angular_variance < angular_kill_threshold,
            "angular_variance": angular_variance,
        },
    }

    # Overall: pass all 4 criteria and not killed
    overall = (
        results["criterion_1_statistical_significance"]["passed"]
        and results["criterion_2_effect_size"]["passed"]
        and results["criterion_3_tecs_mean"]["passed"]
        and results["criterion_4_placebo"]["passed"]
        and not results["kill_gate_angular_variance"]["killed"]
    )
    results["overall_pass"] = overall

    return results


def format_report(
    tecs_results: List,
    test_real_vs_null: TestResult,
    test_edit_vs_placebo: TestResult,
    pass_criteria: dict,
) -> str:
    """Format a human-readable probe results report."""
    lines = [
        "=" * 60,
        "TECA Probe Experiment Results",
        "=" * 60,
        "",
        f"Number of facts: {test_real_vs_null.n}",
        "",
        "--- TECS(real) vs TECS(null-A) ---",
        f"  Mean TECS(real):  {test_real_vs_null.mean_real:.6f}",
        f"  Mean TECS(null):  {test_real_vs_null.mean_null:.6f}",
        f"  t-statistic:      {test_real_vs_null.t_statistic:.4f}",
        f"  p-value:          {test_real_vs_null.p_value:.6f}",
        f"  Cohen's d:        {test_real_vs_null.cohens_d:.4f}  "
        f"[95% CI: {test_real_vs_null.ci_low:.4f}, {test_real_vs_null.ci_high:.4f}]",
        "",
        "--- Placebo Test: TECS(edit layer) vs TECS(non-edit layers) ---",
        f"  Mean TECS(edit):    {test_edit_vs_placebo.mean_real:.6f}",
        f"  Mean TECS(placebo): {test_edit_vs_placebo.mean_null:.6f}",
        f"  t-statistic:        {test_edit_vs_placebo.t_statistic:.4f}",
        f"  p-value:            {test_edit_vs_placebo.p_value:.6f}",
        f"  Cohen's d:          {test_edit_vs_placebo.cohens_d:.4f}",
        "",
        "--- Pass Criteria ---",
    ]
    for k, v in pass_criteria.items():
        if k == "overall_pass":
            continue
        status = "PASS" if v.get("passed", not v.get("killed", False)) else "FAIL"
        lines.append(f"  [{status}] {k}: {v}")

    lines.append("")
    overall = "PASS" if pass_criteria["overall_pass"] else "FAIL"
    lines.append(f">>> OVERALL: {overall} <<<")
    lines.append("=" * 60)

    return "\n".join(lines)
