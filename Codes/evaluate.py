#!/usr/bin/env python3
"""TECA Evaluation and Summary Report Generator.

Reads raw experiment results (JSON) from _Results/ and produces:
- Summary statistics (Cohen's d, CI, p-values) per phase
- Markdown report to _Results/evaluation_report.md
- Pass/fail gates for each experiment

Usage:
    python evaluate.py --config configs/base.yaml
    python evaluate.py --config configs/pilot.yaml --dry-run
    python evaluate.py --results-dir _Results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.config import load_config, config_summary, validate_config


# ---------------------------------------------------------------------------
# Result file discovery
# ---------------------------------------------------------------------------

def find_result_files(results_dir: str) -> Dict[str, str]:
    """Discover all experiment result JSON files in the results directory."""
    files = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        files[name] = path
    # Also check subdirectories
    for path in sorted(glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)):
        name = os.path.relpath(path, results_dir)
        files[name] = path
    return files


def load_result(path: str) -> Optional[Dict[str, Any]]:
    """Load a single result JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase-specific evaluation
# ---------------------------------------------------------------------------

def evaluate_phase_0(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 0: Sanity checks."""
    checks = data.get("checks", {})
    summary = {"phase": 0, "name": "Sanity Checks", "results": {}}

    rome = checks.get("rome_validation", {})
    summary["results"]["rome_validation"] = {
        "efficacy": rome.get("efficacy", "N/A"),
        "gate": "PASS" if rome.get("gate_passed") else "FAIL",
    }

    grad = checks.get("gradient_check", {})
    summary["results"]["gradient_check"] = {
        "gate": "PASS" if grad.get("all_ok") else "FAIL",
    }

    pipeline = checks.get("tecs_pipeline", {})
    summary["results"]["tecs_pipeline"] = {
        "has_nan": pipeline.get("has_nan", "N/A"),
        "all_zero": pipeline.get("all_zero", "N/A"),
        "gate": "PASS" if pipeline.get("pipeline_ok") else "FAIL",
    }

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_1(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 1: Positive controls."""
    exps = data.get("experiments", {})
    summary = {"phase": 1, "name": "Positive Controls", "results": {}}

    # ROME vs self
    rome_self = exps.get("rome_self", {})
    summary["results"]["rome_vs_self"] = {
        "monotonic_decrease": rome_self.get("monotonic_decrease", "N/A"),
        "gate": "PASS" if rome_self.get("monotonic_decrease") else "FAIL",
    }
    if "results" in rome_self:
        for sigma, vals in rome_self["results"].items():
            summary["results"][f"rome_self_sigma_{sigma}"] = vals.get("mean_tecs", "N/A")

    # Toy model
    toy = exps.get("toy_model", {})
    summary["results"]["toy_model"] = {
        "cohens_d": toy.get("cohens_d", "N/A"),
        "p_value": toy.get("p_value", "N/A"),
        "gate": "PASS" if toy.get("gate_passed") else "FAIL",
    }

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_3(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 3: Full-scale core."""
    exps = data.get("experiments", {})
    summary = {"phase": 3, "name": "Full-Scale Core", "results": {}}

    rome = exps.get("rome_editing", {})
    summary["results"]["rome_editing"] = {
        "num_facts": rome.get("num_facts", "N/A"),
        "efficacy": rome.get("efficacy", "N/A"),
    }

    tecs = exps.get("tecs_core", {})
    summary["results"]["tecs_core"] = {
        "cohens_d": tecs.get("cohens_d", "N/A"),
        "p_value": tecs.get("p_value", "N/A"),
        "ci": tecs.get("ci", "N/A"),
        "mean_tecs_real": tecs.get("mean_tecs_real", "N/A"),
        "mean_tecs_null": tecs.get("mean_tecs_null", "N/A"),
    }

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_2(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 2: g_M Quality Analysis."""
    analyses = data.get("analyses", {})
    summary = {"phase": 2, "name": "g_M Quality Analysis", "results": {}}

    wb = analyses.get("within_between", {})
    if wb.get("status") not in ("dry_run", "pending_gpu", None):
        summary["results"]["within_vs_between"] = {
            "within_mean": wb.get("within_fact", {}).get("mean", "N/A"),
            "between_mean": wb.get("between_fact", {}).get("mean", "N/A"),
            "cohens_d": wb.get("statistical_test", {}).get("cohens_d", "N/A"),
            "p_value": wb.get("statistical_test", {}).get("p_value", "N/A"),
        }

    pc1 = analyses.get("pc1_removal", {})
    if pc1.get("status") not in ("dry_run", "pending_gpu", None):
        summary["results"]["pc1_removal"] = {
            "eff_dim_before": pc1.get("pc1_analysis", {}).get("eff_dim_before", "N/A"),
            "eff_dim_after": pc1.get("pc1_analysis", {}).get("eff_dim_after", "N/A"),
            "tecs_before": pc1.get("tecs_comparison", {}).get("abs_tecs_before_mean", "N/A"),
            "tecs_after": pc1.get("tecs_comparison", {}).get("abs_tecs_after_mean", "N/A"),
        }

    ret = analyses.get("retrieval_ablation", {})
    if ret.get("status") not in ("dry_run", "pending_gpu", None):
        summary["results"]["retrieval_ablation"] = {
            "methods": ret.get("methods", "N/A"),
        }

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_4(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 4: Ablation experiments."""
    ablations = data.get("ablations", {})
    summary = {"phase": 4, "name": "Ablation Experiments", "results": {}}

    for axis in ("top_k", "weighting", "loss_function", "scope"):
        abl = ablations.get(axis, {})
        if abl.get("status") not in ("dry_run", "pending_gpu", None):
            tecs_means = abl.get("tecs_means", {})
            if tecs_means:
                values = list(tecs_means.values())
                variation = (max(values) - min(values)) / max(abs(np.mean(values)), 1e-12) * 100
                summary["results"][f"ablation_{axis}"] = {
                    "tecs_means": tecs_means,
                    "variation_pct": round(variation, 2),
                    "gate": "PASS" if variation < 20 else "FAIL",
                }
            else:
                summary["results"][f"ablation_{axis}"] = {"status": abl.get("status", "N/A")}

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_5(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 5: Extended analyses."""
    analyses = data.get("analyses", {})
    summary = {"phase": 5, "name": "Extended Analyses", "results": {}}

    whitening = analyses.get("whitening", {})
    if whitening.get("status") not in ("dry_run", "pending_gpu", None):
        summary["results"]["whitening"] = {
            "cohens_d": whitening.get("cohens_d", "N/A"),
            "p_value": whitening.get("p_value", "N/A"),
        }

    memit = analyses.get("memit", {})
    if memit.get("status") not in ("dry_run", "pending_gpu", None):
        summary["results"]["memit"] = {
            "cross_layer_d": memit.get("cross_layer_cohens_d", "N/A"),
            "matched_layer_d": memit.get("matched_layer_cohens_d", "N/A"),
        }

    summary["overall"] = data.get("status", "unknown")
    return summary


def evaluate_phase_6(data: Dict) -> Dict[str, Any]:
    """Evaluate Phase 6: Cross-model validation."""
    summary = {"phase": 6, "name": "Cross-Model Validation", "results": {}}

    if data.get("status") == "skipped (cross_model.enabled = false)":
        summary["overall"] = "skipped"
        return summary

    tecs = data.get("tecs_core", {})
    if tecs:
        summary["results"]["gptj_tecs"] = {
            "cohens_d": tecs.get("cohens_d", "N/A"),
            "p_value": tecs.get("p_value", "N/A"),
            "mean_tecs_real": tecs.get("mean_tecs_real", "N/A"),
        }

    geom = data.get("subspace_geometry", {})
    if geom:
        summary["results"]["gptj_geometry"] = {
            "eff_dim_edit": geom.get("eff_dim_edit", "N/A"),
            "eff_dim_grad": geom.get("eff_dim_grad", "N/A"),
        }

    summary["overall"] = data.get("status", "unknown")
    return summary


PHASE_EVALUATORS = {
    0: evaluate_phase_0,
    1: evaluate_phase_1,
    2: evaluate_phase_2,
    3: evaluate_phase_3,
    4: evaluate_phase_4,
    5: evaluate_phase_5,
    6: evaluate_phase_6,
}


# ---------------------------------------------------------------------------
# Bootstrap CI (for post-hoc analysis)
# ---------------------------------------------------------------------------

def bootstrap_cohens_d(real: List[float], null: List[float], n_bootstrap: int = 10000,
                       ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """Compute Cohen's d with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    real_arr = np.array(real)
    null_arr = np.array(null)
    diff = real_arr - null_arr
    n = len(diff)

    # Point estimate
    d_point = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # Bootstrap
    d_boot = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diff_b = diff[idx]
        std_b = np.std(diff_b, ddof=1)
        if std_b > 0:
            d_boot.append(np.mean(diff_b) / std_b)

    if not d_boot:
        return {"d": d_point, "ci_low": d_point, "ci_high": d_point}

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(d_boot, 100 * alpha))
    ci_high = float(np.percentile(d_boot, 100 * (1 - alpha)))

    return {"d": float(d_point), "ci_low": ci_low, "ci_high": ci_high, "n_bootstrap": n_bootstrap}


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(
    all_evaluations: Dict[int, Dict],
    cfg: Dict[str, Any],
    results_dir: str,
) -> str:
    """Generate a markdown evaluation report."""
    lines = [
        "# TECA Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **Config summary**: {config_summary(cfg)}",
        f"- **Results directory**: `{results_dir}`",
        "",
        "---",
        "",
    ]

    for phase_id in sorted(all_evaluations.keys()):
        eval_data = all_evaluations[phase_id]
        phase_name = eval_data.get("name", f"Phase {phase_id}")
        overall = eval_data.get("overall", "unknown")

        lines.append(f"## Phase {phase_id}: {phase_name}")
        lines.append("")
        lines.append(f"**Status**: {overall}")
        lines.append("")

        results = eval_data.get("results", {})
        if results:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, val in results.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, float):
                            lines.append(f"| {key}.{sub_key} | {sub_val:.6f} |")
                        else:
                            lines.append(f"| {key}.{sub_key} | {sub_val} |")
                elif isinstance(val, float):
                    lines.append(f"| {key} | {val:.6f} |")
                else:
                    lines.append(f"| {key} | {val} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Summary gate table
    lines.append("## Gate Summary")
    lines.append("")
    lines.append("| Phase | Gate | Status |")
    lines.append("|-------|------|--------|")

    gate_checks = {
        0: [("ROME efficacy >= 75%", lambda e: e.get("results", {}).get("rome_validation", {}).get("gate") == "PASS"),
            ("Gradient check", lambda e: e.get("results", {}).get("gradient_check", {}).get("gate") == "PASS"),
            ("TECS pipeline", lambda e: e.get("results", {}).get("tecs_pipeline", {}).get("gate") == "PASS")],
        1: [("ROME vs self monotonic", lambda e: e.get("results", {}).get("rome_vs_self", {}).get("gate") == "PASS"),
            ("Toy model d > 0.3", lambda e: e.get("results", {}).get("toy_model", {}).get("gate") == "PASS")],
        4: [("Top-k variation < 20%", lambda e: e.get("results", {}).get("ablation_top_k", {}).get("gate") == "PASS"),
            ("Weighting variation < 20%", lambda e: e.get("results", {}).get("ablation_weighting", {}).get("gate") == "PASS"),
            ("Loss fn variation < 20%", lambda e: e.get("results", {}).get("ablation_loss_function", {}).get("gate") == "PASS"),
            ("Scope variation < 20%", lambda e: e.get("results", {}).get("ablation_scope", {}).get("gate") == "PASS")],
    }

    for phase_id in sorted(all_evaluations.keys()):
        eval_data = all_evaluations[phase_id]
        phase_name = eval_data.get("name", f"Phase {phase_id}")
        checks = gate_checks.get(phase_id, [])
        if checks:
            for gate_name, gate_fn in checks:
                status = "PASS" if gate_fn(eval_data) else "FAIL"
                lines.append(f"| {phase_id}: {phase_name} | {gate_name} | {status} |")
        else:
            lines.append(f"| {phase_id}: {phase_name} | N/A | {eval_data.get('overall', 'N/A')} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: Dict[str, Any],
    results_dir: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline."""
    if results_dir is None:
        results_dir = cfg["output"]["results_dir"]

    print("=" * 60)
    print("TECA Evaluation Pipeline")
    print("=" * 60)
    print(f"Results directory: {results_dir}")

    if dry_run:
        print("\n[DRY RUN] Validating evaluation pipeline...")

        # Validate config
        issues = validate_config(cfg)
        if issues:
            for issue in issues:
                print(f"  WARNING: {issue}")

        # Check output directory
        os.makedirs(results_dir, exist_ok=True)

        # Validate report generation with dummy data
        dummy_evals = {
            0: {"name": "Sanity Checks", "overall": "dry_run", "results": {
                "rome_validation": {"efficacy": 0.95, "gate": "PASS"},
                "gradient_check": {"gate": "PASS"},
                "tecs_pipeline": {"gate": "PASS"},
            }},
            1: {"name": "Positive Controls", "overall": "dry_run", "results": {
                "rome_vs_self": {"monotonic_decrease": True, "gate": "PASS"},
                "toy_model": {"cohens_d": 0.8, "p_value": 0.001, "gate": "PASS"},
            }},
        }

        report = generate_report(dummy_evals, cfg, results_dir)
        report_path = os.path.join(results_dir, "evaluation_report_dry_run.md")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Dry-run report written to: {report_path}")
        print("  Evaluation pipeline: OK")

        return {"status": "dry_run_ok", "report_path": report_path}

    # Discover and load results
    result_files = find_result_files(results_dir)
    print(f"\nFound {len(result_files)} result file(s):")
    for name in sorted(result_files.keys()):
        print(f"  - {name}")

    # Load all results
    loaded = {}
    for name, path in result_files.items():
        data = load_result(path)
        if data:
            loaded[name] = data

    # Evaluate each phase from run_experiment.py output (nested under "phases")
    all_evaluations = {}
    for name, data in loaded.items():
        phases = data.get("phases", {})
        for phase_id_str, phase_data in phases.items():
            phase_id = int(phase_id_str)
            if phase_id in PHASE_EVALUATORS:
                eval_result = PHASE_EVALUATORS[phase_id](phase_data)
                all_evaluations[phase_id] = eval_result
                print(f"\n  Evaluated Phase {phase_id}: {eval_result.get('overall', 'unknown')}")

    # Also evaluate standalone result files from individual experiment scripts
    STANDALONE_FILE_MAP = {
        "pc_rome_self.json": (1, "rome_self_standalone"),
        "pc_toy_model.json": (1, "toy_model_standalone"),
        "pc_related_facts.json": (1, "related_facts_standalone"),
        "gm_within_between.json": (2, "within_between_standalone"),
        "gm_pc1_removal.json": (2, "pc1_removal_standalone"),
        "full_tecs_200.json": (3, "tecs_core_standalone"),
        "full_subspace_200.json": (3, "subspace_standalone"),
        "ablation_topk.json": (4, "topk_standalone"),
        "ablation_weighting.json": (4, "weighting_standalone"),
        "ablation_loss.json": (4, "loss_standalone"),
        "ablation_scope.json": (4, "scope_standalone"),
        "ext_whitening_200.json": (5, "whitening_standalone"),
        "ext_memit_200.json": (5, "memit_standalone"),
        "cross_gptj_tecs.json": (6, "gptj_tecs_standalone"),
    }
    for filename, (phase_id, label) in STANDALONE_FILE_MAP.items():
        if filename in loaded and phase_id not in all_evaluations:
            data = loaded[filename]
            summary = {"phase": phase_id, "name": f"Phase {phase_id} ({label})", "results": {}}
            # Extract key metrics from standalone result files
            for key in ("decision", "statistical_test", "tecs_distribution", "tecs",
                        "validation", "pc1_analysis", "tecs_comparison"):
                if key in data and isinstance(data[key], dict):
                    summary["results"][key] = data[key]
            summary["overall"] = data.get("status", data.get("experiment", "standalone"))
            all_evaluations[phase_id] = summary
            print(f"\n  Evaluated Phase {phase_id} from {filename}")

    # Generate report
    report = generate_report(all_evaluations, cfg, results_dir)
    report_path = os.path.join(results_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report written to: {report_path}")

    # Save structured evaluation
    eval_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(eval_path, "w") as f:
        json.dump(all_evaluations, f, indent=2, default=str)
    print(f"  Structured evaluation: {eval_path}")

    return {
        "status": "completed",
        "report_path": report_path,
        "eval_path": eval_path,
        "phases_evaluated": list(all_evaluations.keys()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TECA Evaluation and Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --config configs/pilot.yaml --dry-run
  python evaluate.py --config configs/base.yaml
  python evaluate.py --results-dir _Results
        """,
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config YAML")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate pipeline without processing real results")

    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = args.results_dir or cfg["output"]["results_dir"]

    result = run_evaluation(cfg, results_dir=results_dir, dry_run=args.dry_run)

    if result["status"] == "dry_run_ok":
        print("\nDry run completed successfully.")
    else:
        print(f"\nEvaluation completed. Report: {result.get('report_path')}")


if __name__ == "__main__":
    main()
