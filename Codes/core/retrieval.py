"""Training sample retrieval: BM25 search over OpenWebText or CounterFact paraphrases."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset


# ---------------------------------------------------------------------------
# In-memory BM25 index cache (avoids rebuilding across calls in the same process)
# ---------------------------------------------------------------------------
_BM25_CACHE: Dict[str, dict] = {}


def load_counterfact(
    path: Optional[str] = None,
    num_facts: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Load CounterFact dataset and sample *num_facts* entries.

    Each entry has keys: case_id, pararel_idx, requested_rewrite
    (with prompt, subject, target_new, target_true, relation_id).

    Loads from (in priority order):
    1. A local JSON file at *path*
    2. The original CounterFact JSON from the ROME project data server
    3. HuggingFace datasets (KevinMeng/counterfact or json fallback)

    Returns:
        List of fact dicts with standardized keys:
        {subject, relation, target_old, target_new, prompt, paraphrases, case_id}
    """
    raw = None

    # Priority 1: local file
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)

    # Priority 2: auto-download from ROME project and cache locally
    if raw is None:
        cache_dir = Path("data")
        cache_file = cache_dir / "counterfact.json"
        if cache_file.exists():
            with open(cache_file) as f:
                raw = json.load(f)
        else:
            raw = _download_counterfact(cache_dir, cache_file)

    # Priority 3: HuggingFace dataset
    if raw is None:
        try:
            ds = load_dataset("KevinMeng/counterfact", split="train")
            raw = list(ds)
        except Exception:
            raise RuntimeError(
                "Could not load CounterFact dataset. Please download manually:\n"
                "  wget https://rome.baulab.info/data/dsets/counterfact.json -P data/\n"
                "Or install the dataset: pip install datasets && "
                "python -c \"from datasets import load_dataset; "
                "load_dataset('KevinMeng/counterfact')\""
            )

    import random
    rng = random.Random(seed)
    sampled = rng.sample(raw, min(num_facts, len(raw)))

    facts = []
    for entry in sampled:
        rw = entry.get("requested_rewrite", entry)
        facts.append({
            "case_id": entry.get("case_id", entry.get("id", 0)),
            "subject": rw["subject"],
            "prompt": rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"]
                      else rw["prompt"],
            "target_old": rw.get("target_true", {}).get("str", ""),
            "target_new": rw.get("target_new", {}).get("str", ""),
            "relation_id": rw.get("relation_id", ""),
            "paraphrases": entry.get("paraphrase_prompts", []),
        })
    return facts


def _download_counterfact(cache_dir: Path, cache_file: Path) -> Optional[List[Dict]]:
    """Try to download CounterFact JSON from the ROME project data server."""
    url = "https://rome.baulab.info/data/dsets/counterfact.json"
    try:
        import urllib.request
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CounterFact from {url} ...")
        urllib.request.urlretrieve(url, str(cache_file))
        with open(cache_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: could not download CounterFact from {url}: {e}")
        return None


def retrieve_training_samples_bm25(
    query: str,
    top_k: int = 100,
    corpus_name: str = "openwebtext",
    index_path: Optional[str] = None,
) -> List[Dict]:
    """Retrieve top-k candidate training documents via BM25.

    Uses a three-level cache:
    1. Pre-built pickled index (if index_path is provided and exists)
    2. In-memory cache (if the same corpus was loaded earlier in this process)
    3. Build from streaming corpus shard, then cache in memory

    Args:
        query: The fact-related query string.
        top_k: Number of candidates to retrieve.
        corpus_name: "openwebtext" or other HuggingFace dataset name.
        index_path: Path to a pre-built BM25 index (pickle).

    Returns:
        List of dicts with keys: {text, score, doc_id}
    """
    # Level 1: pre-built pickled index
    if index_path and os.path.exists(index_path):
        return _retrieve_from_prebuilt_index(query, top_k, index_path)

    # Level 2 & 3: in-memory cache or build fresh
    return _retrieve_with_cached_bm25(query, top_k, corpus_name)


def _retrieve_with_cached_bm25(
    query: str, top_k: int, corpus_name: str,
) -> List[Dict]:
    """BM25 retrieval with in-memory caching of the index."""
    global _BM25_CACHE

    if corpus_name not in _BM25_CACHE:
        # Build the index once and cache it
        _BM25_CACHE[corpus_name] = _build_bm25_index(corpus_name)

    cached = _BM25_CACHE[corpus_name]
    bm25 = cached["bm25"]
    corpus_docs = cached["corpus_docs"]

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": corpus_docs[idx]["text"],
            "score": float(scores[idx]),
            "doc_id": corpus_docs[idx]["doc_id"],
        })
    return results


def _build_bm25_index(
    corpus_name: str,
    max_docs: int = 500_000,
) -> dict:
    """Build a BM25 index over a streaming corpus shard."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("pip install rank-bm25")

    print(f"Loading {corpus_name} shard for BM25 indexing...")
    ds = load_dataset(corpus_name, split="train", streaming=True)
    corpus_docs = []
    corpus_texts = []
    for i, doc in enumerate(ds):
        if i >= max_docs:
            break
        text = doc.get("text", "")
        if len(text) > 50:  # skip trivially short docs
            corpus_docs.append({"text": text, "doc_id": i})
            corpus_texts.append(text.lower().split())

    print(f"Building BM25 index over {len(corpus_texts)} documents...")
    bm25 = BM25Okapi(corpus_texts)

    return {"bm25": bm25, "corpus_docs": corpus_docs}


def _retrieve_from_prebuilt_index(
    query: str, top_k: int, index_path: str,
) -> List[Dict]:
    """Retrieve from a pre-built pickled BM25 index.

    Also caches the loaded index in memory for subsequent calls.
    """
    global _BM25_CACHE

    cache_key = f"pickle:{index_path}"
    if cache_key not in _BM25_CACHE:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        _BM25_CACHE[cache_key] = data

    cached = _BM25_CACHE[cache_key]
    bm25 = cached["bm25"]
    corpus_docs = cached["corpus_docs"]

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": corpus_docs[idx]["text"],
            "score": float(scores[idx]),
            "doc_id": corpus_docs[idx].get("doc_id", idx),
        })
    return results


def rank_by_gradient_dot_product(
    candidates: List[Dict],
    gradients: List[torch.Tensor],
    test_gradient: torch.Tensor,
    top_k: int = 10,
) -> List[Tuple[Dict, float]]:
    """Re-rank BM25 candidates by gradient dot product with the test gradient.

    Args:
        candidates: BM25 retrieved docs.
        gradients: Per-sample gradients for each candidate.
        test_gradient: Gradient of test prompt loss.
        top_k: Number of final samples to keep.

    Returns:
        List of (candidate_dict, dot_product_score) tuples, sorted descending.
    """
    test_flat = test_gradient.reshape(-1).float()
    scores = []
    for g in gradients:
        g_flat = g.reshape(-1).float()
        score = torch.dot(test_flat, g_flat).item()
        scores.append(score)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
