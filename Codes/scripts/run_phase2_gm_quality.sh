#!/bin/bash
# Phase 2: g_M Quality Analysis (~1.5 hours)
# 2a: Within/between similarity, 2b: PC1 removal, 2c: Retrieval ablation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODES_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODES_DIR"

echo "============================================"
echo "Phase 2: g_M Quality Analysis"
echo "============================================"

echo ""
echo ">>> 2a: Within-Fact vs Between-Fact Gradient Similarity (~15 min)..."
python -m experiments.gm_quality.within_between_similarity --config configs/phase_2_gm_quality.yaml

echo ""
echo ">>> 2b: PC1 Removal Analysis (~5 min)..."
python -m experiments.gm_quality.pc1_removal --config configs/phase_2_gm_quality.yaml

echo ""
echo ">>> 2c: Retrieval Method Ablation (~1 hour)..."
for method in bm25 tfidf contriever uniform; do
    echo "    Method: ${method}..."
    python -m experiments.gm_quality.retrieval_ablation --config configs/phase_2_gm_quality.yaml --method ${method}
done

echo ""
echo "Phase 2 complete. Check _Results/"
