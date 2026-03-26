#!/bin/bash
# TECA Main Run: Phases 0-5 (200 facts, full analysis)
# Estimated time: ~7 hours on RTX 4090
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CODES_DIR"

echo "============================================"
echo "TECA Main Experiment"
echo "Config: configs/base.yaml"
echo "Phases: 0, 1, 2, 3, 4, 5"
echo "============================================"

# Phase 0: Sanity checks
echo ""
echo ">>> Running Phase 0: Sanity Checks..."
python run_experiment.py --config configs/base.yaml --phase 0

# Phase 1: Positive controls
echo ""
echo ">>> Running Phase 1: Positive Controls..."
python run_experiment.py --config configs/base.yaml --phase 1

# Phase 2: g_M quality analysis
echo ""
echo ">>> Running Phase 2: g_M Quality Analysis..."
python run_experiment.py --config configs/base.yaml --phase 2

# Phase 3: Full-scale core (200 facts)
echo ""
echo ">>> Running Phase 3: Full-Scale Core (200 facts)..."
python run_experiment.py --config configs/base.yaml --phase 3

# Phase 4: Ablation experiments
echo ""
echo ">>> Running Phase 4: Ablation Experiments..."
python run_experiment.py --config configs/base.yaml --phase 4

# Phase 5: Extended analyses
echo ""
echo ">>> Running Phase 5: Extended Analyses..."
python run_experiment.py --config configs/base.yaml --phase 5

# Evaluate
echo ""
echo ">>> Generating evaluation report..."
python evaluate.py --config configs/base.yaml

echo ""
echo "============================================"
echo "Main experiment complete. Check _Results/"
echo "============================================"
