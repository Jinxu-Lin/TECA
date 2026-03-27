#!/bin/bash
# Phase 1: Positive Control Experiments (~35 min)
# 1a: ROME vs Self, 1b: Toy Model, 1c: Related Facts
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 1: Positive Control Experiments"
echo "============================================"

# Option A: Run via unified runner (all sub-experiments)
# python run_experiment.py --config configs/phase_1_positive_control.yaml --phase 1

# Option B: Run individual scripts (preferred for GPU execution)
echo ""
echo ">>> 1a: ROME vs Self (~5 min)..."
python -m experiments.positive_control.rome_self_check --config configs/phase_1_positive_control.yaml

echo ""
echo ">>> 1b: Toy Model (~10 min, CPU)..."
python -m experiments.positive_control.toy_model_tecs --config configs/phase_1_positive_control.yaml

echo ""
echo ">>> 1c: Related Facts (~20 min)..."
python -m experiments.positive_control.related_facts_tecs --config configs/phase_1_positive_control.yaml

echo ""
echo "Phase 1 complete. Check _Results/"
echo "CRITICAL GATE: If toy model d < 0.3, STOP and escalate."
