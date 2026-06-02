# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Benchmark pruners for evalscope.

A pruner selects a representative subset of benchmark items from pre-computed
model scores so that evaluating only the subset gives a reliable go/no-go
signal equivalent to running the full benchmark.

Quick start
-----------
    import json, numpy as np
    from evalscope.pruners import SDSPruner

    # Load your pre-computed score matrix (n_items x n_models)
    score_matrix = np.load("scores.npy")           # or build from JSONL files

    pruner = SDSPruner(n_bands=10)
    indices = pruner.prune(score_matrix, n=35)     # 35 items from e.g. LCB v5

    print(f"Selected {len(indices)} items: {indices[:5]} ...")

Or from the command line::

    evalscope prune \\
        --scores path/to/score_matrix.jsonl \\
        --n 35 \\
        --output selected_indices.json
"""

from .base import BasePruner
from .noise_filter import estimate_noise_weights
from .sds_pruner import SDSPruner

__all__ = [
    'BasePruner',
    'SDSPruner',
    'estimate_noise_weights',
]
