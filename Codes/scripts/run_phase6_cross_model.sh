#!/bin/bash
# Phase 6: Cross-Model Validation (~3 hours)
# GPT-J-6B: ROME + TDA + TECS + positive control
# Run ONLY after Phases 1-5 succeed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 6: Cross-Model Validation (GPT-J-6B)"
echo "============================================"

echo ""
echo ">>> 6a: GPT-J ROME editing on 100 facts..."
python -m experiments.cross_model.gptj_rome --config configs/phase_6_cross_model.yaml

echo ""
echo ">>> 6b: GPT-J TDA gradients..."
python -m experiments.cross_model.gptj_tda --config configs/phase_6_cross_model.yaml

echo ""
echo ">>> 6c: GPT-J TECS + subspace geometry..."
python -m experiments.cross_model.gptj_tecs --config configs/phase_6_cross_model.yaml

echo ""
echo ">>> 6d: GPT-J positive control..."
python -m experiments.cross_model.gptj_positive_control --config configs/phase_6_cross_model.yaml

echo ""
echo "Phase 6 complete. Check _Results/"
