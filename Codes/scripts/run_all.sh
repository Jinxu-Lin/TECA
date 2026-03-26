#!/bin/bash
# TECA Full Run: All Phases 0-7 (including cross-model + visualization)
# Estimated time: ~10 hours on RTX 4090
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CODES_DIR"

echo "============================================"
echo "TECA Full Experiment Suite"
echo "Config: configs/base.yaml (with cross-model enabled)"
echo "Phases: 0-7"
echo "============================================"

# Phases 0-5: Main experiments
echo ""
echo ">>> Running Phases 0-5 (main experiments)..."
for phase in 0 1 2 3 4 5; do
    echo ""
    echo ">>> Phase ${phase}..."
    python run_experiment.py --config configs/base.yaml --phase ${phase}
done

# Phase 6: Cross-model (GPT-J-6B)
echo ""
echo ">>> Running Phase 6: Cross-Model Validation (GPT-J-6B)..."
python run_experiment.py --config configs/phase_6_cross_model.yaml --phase 6

# Phase 7: Visualization
echo ""
echo ">>> Running Phase 7: Visualization..."
python run_experiment.py --config configs/base.yaml --phase 7

# Final evaluation
echo ""
echo ">>> Generating final evaluation report..."
python evaluate.py --config configs/base.yaml

echo ""
echo "============================================"
echo "Full experiment suite complete. Check _Results/"
echo "============================================"
