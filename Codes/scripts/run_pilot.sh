#!/bin/bash
# TECA Pilot Run: Phase 0 (sanity) + Phase 1 (positive control) + Phase 3 (core, 50 facts)
# Estimated time: ~1 hour on RTX 4090
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CODES_DIR"

echo "============================================"
echo "TECA Pilot Experiment"
echo "Config: configs/pilot.yaml"
echo "Phases: 0, 1, 3"
echo "============================================"

# Phase 0: Sanity checks
echo ""
echo ">>> Running Phase 0: Sanity Checks..."
python run_experiment.py --config configs/pilot.yaml --phase 0

# Phase 1: Positive controls (gate check)
echo ""
echo ">>> Running Phase 1: Positive Controls..."
python run_experiment.py --config configs/pilot.yaml --phase 1

# Phase 3: Core TECS (50 facts)
echo ""
echo ">>> Running Phase 3: Full-Scale Core (50 facts)..."
python run_experiment.py --config configs/pilot.yaml --phase 3

# Evaluate
echo ""
echo ">>> Generating evaluation report..."
python evaluate.py --config configs/pilot.yaml

echo ""
echo "============================================"
echo "Pilot complete. Check _Results/pilot/"
echo "============================================"
