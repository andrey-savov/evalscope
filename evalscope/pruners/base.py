# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Base class for benchmark pruners.

A pruner selects a representative subset of benchmark items from a pre-computed
score matrix so that running only the subset gives a go/no-go signal equivalent
to running the full benchmark.

Usage pattern
-------------
1. You have already run (or were given) per-sample scores from K models on N items.
2. Build a 2-D score matrix M of shape (N, K).
3. Call pruner.prune(score_matrix, n) to get the indices of the selected items.
4. Pass those indices to your evalscope dataset loader via the ``subset_indices``
   argument so only those items are evaluated on the new model.
"""

from __future__ import annotations

import abc
from typing import Sequence


class BasePruner(abc.ABC):
    """Abstract base class for benchmark pruners.

    Subclasses must implement :meth:`prune`.
    """

    @abc.abstractmethod
    def prune(
        self,
        score_matrix: "numpy.ndarray",  # noqa: F821 (numpy imported in subclass)
        n: int,
        *,
        metadata: Sequence[dict] | None = None,
    ) -> list[int]:
        """Select *n* item indices from *score_matrix*.

        Parameters
        ----------
        score_matrix:
            Float array of shape ``(n_items, n_models)`` containing per-item
            scores (typically 0/1 for pass/fail benchmarks).  Scores from the
            training models — i.e. models whose outputs were used to calibrate
            the pruner.  The new (test) model is *not* represented here.
        n:
            Target subset size.  If ``n >= n_items`` all items are returned.
        metadata:
            Optional list of per-item dicts (length ``n_items``).  Subclasses
            may use fields such as ``"input_tokens"`` as secondary sort keys.

        Returns
        -------
        list[int]
            Sorted list of row indices into *score_matrix* of length ``n``
            (or ``n_items`` when ``n >= n_items``).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
