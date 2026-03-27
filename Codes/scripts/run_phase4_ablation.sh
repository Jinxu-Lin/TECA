#!/bin/bash
# Phase 4: Ablation Experiments (~1 hour)
# Top-k, weighting, loss function, gradient scope
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 4: Ablation Experiments"
echo "============================================"

echo ""
echo ">>> 4a: Top-k ablation (k in {5, 10, 20, 50})..."
python -m experiments.ablation.topk_ablation --config configs/phase_4_ablation.yaml

echo ""
echo ">>> 4b: Weighting scheme ablation (BM25 / uniform / TF-IDF)..."
python -m experiments.ablation.weighting_ablation --config configs/phase_4_ablation.yaml

echo ""
echo ">>> 4c: Loss function ablation (object CE / full CE / margin)..."
python -m experiments.ablation.loss_ablation --config configs/phase_4_ablation.yaml

echo ""
echo ">>> 4d: Gradient scope ablation (single-layer vs multi-layer)..."
python -m experiments.ablation.scope_ablation --config configs/phase_4_ablation.yaml

echo ""
echo "Phase 4 complete. Check _Results/"
