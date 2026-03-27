#!/bin/bash
# TECA Full Run: All Phases 0-7
# Estimated time: ~10 hours on RTX 4090
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "TECA Full Experiment Suite"
echo "Phases: 0-7 (all experiments)"
echo "Estimated time: ~10 hours on RTX 4090"
echo "============================================"

# P0 Priority: Phases 0-3 (~4 hours)
echo ""
echo ">>> P0 Priority: Phases 0-3..."
bash scripts/run_all_p0.sh

# P1 Priority: Phases 4-5 (~2.5 hours)
echo ""
echo ">>> Phase 4: Ablation Experiments..."
bash scripts/run_phase4_ablation.sh

echo ""
echo ">>> Phase 5: Extended Analyses..."
bash scripts/run_phase5_extended.sh

# P2 Priority: Phase 6 (~3 hours)
echo ""
echo ">>> Phase 6: Cross-Model Validation..."
bash scripts/run_phase6_cross_model.sh

# Phase 7: Visualization
echo ""
echo ">>> Phase 7: Visualization..."
python run_experiment.py --config configs/phase_7_visualization.yaml --phase 7

# Final evaluation
echo ""
echo ">>> Generating final evaluation report..."
python evaluate.py --config configs/base.yaml

echo ""
echo "============================================"
echo "Full experiment suite complete. Check _Results/"
echo "============================================"
