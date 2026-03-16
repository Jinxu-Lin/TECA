"""Build and pickle a BM25 index for offline use.

Run this once before the main probe experiment to avoid rebuilding the index
for every fact. The pickled index is loaded by retrieval.py when passed via
--bm25_index.

Usage:
    python -m experiments.build_bm25_index \
        --corpus openwebtext \
        --max_docs 500000 \
        --output data/bm25_index_owt500k.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retrieval import _build_bm25_index


def main():
    parser = argparse.ArgumentParser(description="Build and pickle a BM25 index")
    parser.add_argument("--corpus", type=str, default="openwebtext",
                        help="HuggingFace dataset name for the corpus")
    parser.add_argument("--max_docs", type=int, default=500_000,
                        help="Maximum number of documents to index")
    parser.add_argument("--output", type=str, default="data/bm25_index_owt500k.pkl",
                        help="Output path for the pickled index")
    args = parser.parse_args()

    print("=" * 50)
    print("BM25 Index Builder")
    print("=" * 50)
    print(f"  Corpus: {args.corpus}")
    print(f"  Max docs: {args.max_docs}")
    print(f"  Output: {args.output}")

    t0 = time.time()

    # Build the index
    index_data = _build_bm25_index(args.corpus, max_docs=args.max_docs)

    # Save to pickle
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(index_data, f)

    elapsed = time.time() - t0
    n_docs = len(index_data["corpus_docs"])
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"  Indexed {n_docs} documents")
    print(f"  Saved to {args.output} ({file_size_mb:.1f} MB)")
    print(f"\nUsage: python -m experiments.probe_main --bm25_index {args.output}")


if __name__ == "__main__":
    main()
