# Copyright (c) Alibaba, Inc. and its affiliates.
"""
LLM judge noise filter for benchmarks with non-deterministic scoring.

When scores come from an LLM judge (e.g. AA-LCR), each score is a noisy
observation of the item's true difficulty.  An item with stable scores across
models (all high or all low) likely reflects a genuine easy/hard item.  An
item whose scores are inconsistent with a model's overall rank may instead
reflect judge noise rather than a real capability difference.

This module provides a lightweight noise-detection heuristic: for each item,
compute how far each model's score deviates from its expected score given its
average rank.  Items with large unexplained deviations are flagged as
potentially noisy and down-weighted in the discrimination signal passed to
SDSPruner.

Usage
-----
    from evalscope.pruners.noise_filter import estimate_noise_weights
    from evalscope.pruners.sds_pruner import SDSPruner
    import numpy as np

    # score_matrix: (n_items, n_models) with binary 0/1 scores
    weights = estimate_noise_weights(score_matrix, threshold=1.5)
    # weights[i] ~= 1.0 for reliable items, < 1.0 for suspected noisy items

    # Optionally pass to SDSPruner by reweighting the score_matrix:
    reweighted = score_matrix * weights[:, None]
    pruner = SDSPruner()
    indices = pruner.prune(reweighted, n=20)
"""

from __future__ import annotations

import numpy as np


def estimate_noise_weights(
    score_matrix: np.ndarray,
    threshold: float = 1.5,
) -> np.ndarray:
    """Estimate per-item reliability weights based on score consistency.

    For each item *i* and each model *m*, compute the *residual*:

        residual[i, m] = score[i, m] - mean_score[m]

    across all items.  Items where any model's residual exceeds *threshold*
    standard deviations receive a reduced weight.

    Parameters
    ----------
    score_matrix:
        Float array of shape ``(n_items, n_models)``.
    threshold:
        Z-score threshold above which a model's score on an item is considered
        a potential judge-noise artefact.  Default 1.5 is deliberately
        conservative — we prefer to flag items that are genuinely inconsistent
        rather than miss noise.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n_items,)`` with values in ``(0, 1]``.
        Reliable items have weight 1.0; suspected noisy items have weight 0.5.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    n_items, n_models = score_matrix.shape

    if n_models < 2:
        return np.ones(n_items)

    # Per-item, per-model residuals relative to each model's column mean.
    col_mean = score_matrix.mean(axis=0)      # (n_models,)
    col_std  = score_matrix.std(axis=0)       # (n_models,)
    col_std  = np.where(col_std == 0, 1.0, col_std)  # avoid divide-by-zero

    z_scores = np.abs((score_matrix - col_mean) / col_std)   # (n_items, n_models)

    # An item is flagged if ANY model's score is an outlier.
    noisy_mask = z_scores.max(axis=1) > threshold             # (n_items,)

    weights = np.where(noisy_mask, 0.5, 1.0).astype(float)
    return weights
