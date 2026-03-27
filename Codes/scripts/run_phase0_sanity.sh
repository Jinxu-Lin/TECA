#!/bin/bash
# Phase 0: Sanity Checks (~10 min)
# ROME validation, gradient check, TECS pipeline check
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 0: Sanity Checks"
echo "============================================"

python run_experiment.py --config configs/phase_0_sanity.yaml --phase 0

echo ""
echo "Phase 0 complete. Check _Results/"
