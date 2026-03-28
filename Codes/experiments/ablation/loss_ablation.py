#!/usr/bin/env python3
"""Phase 4: Loss Function Ablation.

Compare object-token CE / full-sequence CE / margin loss for gradient computation.
Expected: TECS variation < 20% across loss definitions.

Usage:
    python -m experiments.ablation.loss_ablation --config configs/phase_4_ablation.yaml
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat,
    load_counterfact_facts, get_results_dir,
)
from core.config import load_config
from core.model_utils import get_mlp_proj_param


def compute_gradient_with_loss(model, tokenizer, text, layer_idx, device, loss_type="full_sequence_ce"):
    """Compute gradient with a specified loss function."""
    param = get_mlp_proj_param(model, layer_idx)
    model.zero_grad()
    backup = param.requires_grad
    param.requires_grad_(True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

    if loss_type == "full_sequence_ce":
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    elif loss_type == "object_token_ce":
        # Only compute loss on the last few tokens (object position)
        outputs = model(**inputs)
        logits = outputs.logits
        n_tokens = min(3, inputs["input_ids"].shape[1] - 1)
        if n_tokens < 1:
            param.requires_grad_(backup)
            model.zero_grad()
            return None

        target_logits = logits[0, -(n_tokens + 1):-1, :]
        target_ids = inputs["input_ids"][0, -n_tokens:]
        loss = F.cross_entropy(target_logits, target_ids)

    elif loss_type == "margin":
        # Margin loss: log P(correct) - log P(second best)
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits[0, :-1, :]  # (seq-1, vocab)
        targets = inputs["input_ids"][0, 1:]  # (seq-1,)

        log_probs = F.log_softmax(logits, dim=-1)
        correct_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Second best: mask out correct token, get max
        masked_logits = logits.clone()
        masked_logits.scatter_(1, targets.unsqueeze(1), float('-inf'))
        second_log_probs = F.log_softmax(masked_logits, dim=-1).max(dim=-1).values

        margin = correct_log_probs - second_log_probs
        loss = -margin.mean()  # Minimize negative margin

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss.backward()
    grad = param.grad.detach().clone().cpu()
    param.requires_grad_(backup)
    model.zero_grad()

    return grad


def run_loss_ablation(cfg: dict) -> dict:
    """Run loss function ablation."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

    abl_cfg = cfg.get("ablation", {})
    loss_functions = abl_cfg.get("loss_functions", ["object_token_ce", "full_sequence_ce", "margin"])

    data_cfg = cfg.get("data", {})
    counterfact_path = data_cfg.get("counterfact_path", "data/counterfact.json")
    num_facts = data_cfg.get("num_facts", 200)
    model_name = cfg.get("model", {}).get("name", "gpt2-xl")
    device = cfg.get("model", {}).get("device", "cuda")
    edit_layer = cfg.get("model", {}).get("edit_layer") or 17

    retrieval_cfg = cfg.get("retrieval", {})
    top_k_candidates = retrieval_cfg.get("top_k_candidates", 100)
    top_k_gradient = retrieval_cfg.get("top_k_gradient", 10)
    index_path = retrieval_cfg.get("index_path")

    print("=" * 60)
    print("Phase 4: Loss Function Ablation")
    print("=" * 60)
    print(f"  Loss functions: {loss_functions}")

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    loss_results = {lf: [] for lf in loss_functions}

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\nProcessing {len(facts)} facts...")
    for i, fact in enumerate(facts):
        try:
            edit_result = compute_rome_edit(
                model, tokenizer,
                subject=fact["subject"],
                prompt=fact["prompt"],
                target_new=fact["target_new"],
                target_old=fact["target_old"],
                edit_layer=edit_layer,
                device=device,
            )
            delta = edit_result.delta_weight.cpu()

            query = f"{fact['subject']} {fact['target_old']}"
            retrieved = retrieve_training_samples_bm25(
                query, top_k=top_k_candidates, index_path=index_path,
            )
            training_texts = [r["text"][:512] for r in retrieved[:top_k_gradient]]
            weights = [r["score"] for r in retrieved[:top_k_gradient]]

            for loss_type in loss_functions:
                grads = []
                for text in training_texts:
                    g = compute_gradient_with_loss(
                        model, tokenizer, text, edit_layer, device, loss_type,
                    )
                    if g is not None:
                        grads.append(g)

                if not grads:
                    continue

                w = torch.tensor(weights[:len(grads)], dtype=torch.float32)
                w = w / w.sum()
                stacked = torch.stack(grads)
                g_M = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
                tecs = cosine_similarity_flat(delta, g_M)
                loss_results[loss_type].append(tecs)

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    summary = {}
    means = []
    for lf in loss_functions:
        arr = np.array(loss_results[lf])
        summary[lf] = {
            "n": len(arr),
            "mean": float(arr.mean()) if len(arr) > 0 else 0,
            "std": float(arr.std()) if len(arr) > 0 else 0,
        }
        if len(arr) > 0:
            means.append(float(arr.mean()))
        print(f"  {lf}: TECS={arr.mean():.6f} (n={len(arr)})" if len(arr) > 0 else f"  {lf}: no results")

    mean_tecs = np.mean(means) if means else 0
    max_var = max(abs(m - mean_tecs) / abs(mean_tecs) for m in means) if abs(mean_tecs) > 1e-12 and means else 0

    elapsed = time.time() - start_time

    results = {
        "experiment": "ablation_loss",
        "phase": "4",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {"loss_functions": loss_functions, "num_facts": num_facts, "seed": seed},
        "per_loss_results": summary,
        "robustness": {
            "mean_tecs_across_losses": float(mean_tecs),
            "max_relative_variation": float(max_var),
            "robust": max_var < 0.20,
        },
    }

    save_results(results, os.path.join(results_dir, "ablation_loss.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4: Loss Ablation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_loss_ablation(cfg)


if __name__ == "__main__":
    main()
