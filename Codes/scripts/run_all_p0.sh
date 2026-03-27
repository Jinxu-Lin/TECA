#!/bin/bash
# TECA P0 Priority: Phases 0, 1, 2, 3 (~4 hours)
# Critical path experiments that must complete first.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "TECA P0 Priority Experiments"
echo "Phases: 0 (sanity) + 1 (positive control) + 2 (g_M quality) + 3 (full scale)"
echo "Estimated time: ~4 hours on RTX 4090"
echo "============================================"

# Phase 0: Sanity (~10 min)
echo ""
echo ">>> Phase 0: Sanity Checks..."
bash scripts/run_phase0_sanity.sh

# Phase 1: Positive Control (~35 min)
echo ""
echo ">>> Phase 1: Positive Controls..."
bash scripts/run_phase1_positive_control.sh

# GATE CHECK: verify toy model passed
if [ -f "_Results/pc_toy_model.json" ]; then
    GATE=$(python -c "import json; d=json.load(open('_Results/pc_toy_model.json')); print('PASS' if d.get('decision',{}).get('pass_d_03') else 'FAIL')" 2>/dev/null || echo "UNKNOWN")
    if [ "$GATE" = "FAIL" ]; then
        echo ""
        echo "*** CRITICAL GATE FAILURE ***"
        echo "Toy model TECS d < 0.3 -- TECS metric may be flawed."
        echo "Stopping. Escalate to design phase."
        exit 1
    fi
    echo "  Toy model gate: $GATE"
fi

# Phase 2: g_M Quality (~1.5 hours)
echo ""
echo ">>> Phase 2: g_M Quality Analysis..."
bash scripts/run_phase2_gm_quality.sh

# Phase 3: Full Scale (~2 hours)
echo ""
echo ">>> Phase 3: Full-Scale Core..."
bash scripts/run_phase3_full_scale.sh

# Evaluate
echo ""
echo ">>> Generating P0 evaluation report..."
python evaluate.py --config configs/base.yaml

echo ""
echo "============================================"
echo "P0 experiments complete. Check _Results/"
echo "Next: run_phase4_ablation.sh, run_phase5_extended.sh"
echo "============================================"
