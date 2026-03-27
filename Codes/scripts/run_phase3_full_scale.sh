#!/bin/bash
# Phase 3: Full-Scale Core Experiments (~2 hours)
# ROME 200 facts + TDA gradients + TECS core + Subspace geometry
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 3: Full-Scale Core Experiments"
echo "============================================"

echo ""
echo ">>> 3a: ROME editing on 200 facts..."
python -m experiments.full_scale.rome_200 --config configs/phase_3_full_scale.yaml

echo ""
echo ">>> 3b: TDA gradient computation for 200 facts..."
python -m experiments.full_scale.tda_gradients_200 --config configs/phase_3_full_scale.yaml

echo ""
echo ">>> 3c: Core TECS measurement + 5 null baselines..."
python -m experiments.full_scale.tecs_core_200 --config configs/phase_3_full_scale.yaml

echo ""
echo ">>> 3d: Subspace geometry analysis..."
python -m experiments.full_scale.subspace_geometry_200 --config configs/phase_3_full_scale.yaml

echo ""
echo "Phase 3 complete. Check _Results/"
