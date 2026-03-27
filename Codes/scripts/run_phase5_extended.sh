#!/bin/bash
# Phase 5: Extended Analyses (~1.5 hours)
# Whitening decomposition (H6), MEMIT comparison
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 5: Extended Analyses"
echo "============================================"

echo ""
echo ">>> 5a: Whitening decomposition (200 facts)..."
python -m experiments.full_scale.whitening_200 --config configs/phase_5_extended.yaml

echo ""
echo ">>> 5b: MEMIT comparison (200 facts, layers 13-17)..."
python -m experiments.full_scale.memit_200 --config configs/phase_5_extended.yaml

echo ""
echo "Phase 5 complete. Check _Results/"
