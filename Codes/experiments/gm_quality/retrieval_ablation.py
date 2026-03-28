#!/usr/bin/env python3
"""Phase 2c: Retrieval Method Ablation.

Compare attribution quality across retrieval methods:
  BM25 (baseline), TF-IDF, Contriever (dense), Uniform (random).
For each method: compute g_M, measure eff-dim, re-compute TECS.

Usage:
    python -m experiments.gm_quality.retrieval_ablation --config configs/phase_2_gm_quality.yaml --method bm25
    python -m experiments.gm_quality.retrieval_ablation --config configs/phase_2_gm_quality.yaml --method tfidf
    python -m experiments.gm_quality.retrieval_ablation --config configs/phase_2_gm_quality.yaml --method contriever
    python -m experiments.gm_quality.retrieval_ablation --config configs/phase_2_gm_quality.yaml --method uniform
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.common import (
    set_seed, save_results, cosine_similarity_flat, cohens_d,
    bootstrap_ci, paired_test, load_counterfact_facts, get_results_dir,
)
from core.config import load_config


def compute_effective_dimensionality(S):
    """Effective dimensionality from singular values via eigenvalue entropy."""
    s2 = S ** 2
    p = s2 / s2.sum()
    p = p[p > 1e-12]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def retrieve_tfidf(query: str, corpus_docs: List[Dict], tfidf_vectorizer, tfidf_matrix,
                   top_k: int = 100) -> List[Dict]:
    """Retrieve using TF-IDF similarity."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    query_vec = tfidf_vectorizer.transform([query])
    scores = sk_cosine(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [
        {"text": corpus_docs[i]["text"], "score": float(scores[i]), "doc_id": corpus_docs[i].get("doc_id", i)}
        for i in top_indices
    ]


def retrieve_uniform(corpus_docs: List[Dict], top_k: int = 100, rng=None) -> List[Dict]:
    """Retrieve uniformly random documents."""
    if rng is None:
        rng = np.random.RandomState(42)
    indices = rng.choice(len(corpus_docs), size=min(top_k, len(corpus_docs)), replace=False)
    return [
        {"text": corpus_docs[i]["text"], "score": 1.0, "doc_id": corpus_docs[i].get("doc_id", i)}
        for i in indices
    ]


def retrieve_contriever(query: str, corpus_docs: List[Dict], contriever_model,
                        contriever_tokenizer, corpus_embeddings, top_k: int = 100,
                        device: str = "cpu") -> List[Dict]:
    """Retrieve using Contriever dense embeddings."""
    import torch.nn.functional as F

    # Encode query
    inputs = contriever_tokenizer(query, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = contriever_model(**inputs)
        q_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        q_emb = F.normalize(q_emb, dim=1)

    # Compute similarities
    scores = (q_emb @ corpus_embeddings.T).squeeze(0).cpu().numpy()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [
        {"text": corpus_docs[i]["text"], "score": float(scores[i]), "doc_id": corpus_docs[i].get("doc_id", i)}
        for i in top_indices
    ]


def build_retrieval_index(method: str, corpus_docs: List[Dict], device: str = "cpu"):
    """Build retrieval index for the specified method. Returns method-specific objects."""
    if method == "bm25":
        # BM25 handled by core.retrieval
        return {}

    elif method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [doc["text"] for doc in corpus_docs]
        vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return {"vectorizer": vectorizer, "matrix": matrix}

    elif method == "contriever":
        from transformers import AutoModel, AutoTokenizer as AT
        import torch.nn.functional as F

        print("  Loading Contriever model...")
        ct_tokenizer = AT.from_pretrained("facebook/contriever")
        ct_model = AutoModel.from_pretrained("facebook/contriever").to(device).eval()

        # Encode corpus in batches
        print("  Encoding corpus documents...")
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(corpus_docs), batch_size):
            batch_texts = [doc["text"][:512] for doc in corpus_docs[i:i + batch_size]]
            inputs = ct_tokenizer(batch_texts, return_tensors="pt", truncation=True,
                                  max_length=256, padding=True).to(device)
            with torch.no_grad():
                outputs = ct_model(**inputs)
                embs = outputs.last_hidden_state[:, 0, :]
                embs = F.normalize(embs, dim=1)
                all_embeddings.append(embs.cpu())

            if (i + batch_size) % 1000 == 0:
                print(f"    Encoded {min(i + batch_size, len(corpus_docs))}/{len(corpus_docs)}")

        corpus_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        return {"model": ct_model, "tokenizer": ct_tokenizer, "embeddings": corpus_embeddings}

    elif method == "uniform":
        return {}

    else:
        raise ValueError(f"Unknown retrieval method: {method}")


def run_retrieval_ablation(cfg: dict, method: str) -> dict:
    """Run retrieval method ablation for a single method."""
    seed = cfg.get("seed", 42)
    set_seed(seed)

    results_dir = get_results_dir(cfg)
    os.makedirs(results_dir, exist_ok=True)

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
    print(f"Phase 2c: Retrieval Ablation — Method: {method.upper()}")
    print("=" * 60)

    start_time = time.time()

    from core.model_utils import load_model_and_tokenizer
    from core.rome_utils import compute_rome_edit
    from core.gradient_utils import compute_aggregated_gradient
    from core.retrieval import retrieve_training_samples_bm25

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    facts = load_counterfact_facts(counterfact_path, num_facts=num_facts, seed=seed)

    # Build corpus for non-BM25 methods
    corpus_docs = None
    retrieval_index = {}
    if method in ("tfidf", "contriever", "uniform"):
        # Load corpus subset
        print("  Loading corpus for retrieval index...")
        from core.retrieval import _build_bm25_index
        corpus_name = retrieval_cfg.get("corpus", "openwebtext")
        max_docs = min(retrieval_cfg.get("max_docs", 500000), 100000)  # Cap for ablation

        from datasets import load_dataset
        ds = load_dataset(corpus_name, split="train", streaming=True)
        corpus_docs = []
        for i, doc in enumerate(ds):
            if i >= max_docs:
                break
            text = doc.get("text", "")
            if len(text) > 50:
                corpus_docs.append({"text": text, "doc_id": i})

        print(f"  Corpus size: {len(corpus_docs)}")
        retrieval_index = build_retrieval_index(method, corpus_docs, device=device)

    rng = np.random.RandomState(seed)

    # Collect ROME deltas and gradients
    deltas = []
    gradients = []
    per_fact = []

    print(f"\nComputing for {len(facts)} facts...")
    for i, fact in enumerate(facts):
        try:
            # ROME edit (same for all methods)
            edit_result = compute_rome_edit(
                model, tokenizer,
                subject=fact["subject"],
                prompt=fact["prompt"],
                target_new=fact["target_new"],
                target_old=fact["target_old"],
                edit_layer=edit_layer,
                device=device,
            )

            # Retrieval
            query = f"{fact['subject']} {fact['target_old']}"

            if method == "bm25":
                retrieved = retrieve_training_samples_bm25(
                    query, top_k=top_k_candidates, index_path=index_path,
                )
            elif method == "tfidf":
                retrieved = retrieve_tfidf(
                    query, corpus_docs, retrieval_index["vectorizer"],
                    retrieval_index["matrix"], top_k=top_k_candidates,
                )
            elif method == "contriever":
                retrieved = retrieve_contriever(
                    query, corpus_docs, retrieval_index["model"],
                    retrieval_index["tokenizer"], retrieval_index["embeddings"],
                    top_k=top_k_candidates, device=device,
                )
            elif method == "uniform":
                retrieved = retrieve_uniform(corpus_docs, top_k=top_k_candidates, rng=rng)
            else:
                raise ValueError(f"Unknown method: {method}")

            training_texts = [r["text"][:512] for r in retrieved[:top_k_gradient]]
            weights = [r["score"] for r in retrieved[:top_k_gradient]] if method != "uniform" else None

            g_M = compute_aggregated_gradient(
                model, tokenizer, fact["prompt"],
                training_texts, edit_layer, device=device,
                top_k=top_k_gradient, weights=weights,
            )

            deltas.append(edit_result.delta_weight.cpu())
            gradients.append(g_M.cpu())

            tecs = cosine_similarity_flat(edit_result.delta_weight.cpu(), g_M.cpu())
            per_fact.append({"case_id": fact["case_id"], "tecs": tecs})

        except Exception as e:
            print(f"  [WARN] Fact {i}: {e}")
            continue

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(facts)}]")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    n_valid = len(deltas)
    print(f"  {n_valid} valid facts")

    # Compute effective dimensionality
    G_flat = torch.stack([g.reshape(-1).float() for g in gradients])
    G_centered = G_flat - G_flat.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd_lowrank(G_centered, q=min(n_valid, 100), niter=5)
    eff_dim = compute_effective_dimensionality(S)

    # TECS statistics
    tecs_values = np.array([pf["tecs"] for pf in per_fact])

    elapsed = time.time() - start_time

    results = {
        "experiment": f"gm_retrieval_{method}",
        "phase": "2c",
        "method": method,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed,
        "config": {
            "model": model_name,
            "edit_layer": edit_layer,
            "num_facts": n_valid,
            "retrieval_method": method,
            "top_k_candidates": top_k_candidates,
            "top_k_gradient": top_k_gradient,
            "seed": seed,
        },
        "tecs": {
            "mean": float(tecs_values.mean()),
            "std": float(tecs_values.std()),
            "median": float(np.median(tecs_values)),
        },
        "effective_dimensionality": eff_dim,
        "top_5_singular_values": S[:5].tolist(),
        "per_fact_results": per_fact,
    }

    print(f"\n{'=' * 60}")
    print(f"  Retrieval Ablation ({method.upper()}):")
    print(f"  TECS mean: {tecs_values.mean():.6f}")
    print(f"  Eff-dim: {eff_dim:.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    save_results(results, os.path.join(results_dir, f"gm_retrieval_{method}.json"))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2c: Retrieval Method Ablation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["bm25", "tfidf", "contriever", "uniform"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_retrieval_ablation(cfg, args.method)


if __name__ == "__main__":
    main()
