# Copyright (c) Alibaba, Inc. and its affiliates.
"""
AA-LCR Pruned Adapter.

Registers ``aa_lcr_pruned`` — an SDS-selected subset of AA-LCR, following the
same pattern as ``live_code_bench_pruned``.

Quick start
-----------
Step 1 — build the subset once (with noise filter for judge-scored benchmark)::

    evalscope prune --scores aa_lcr_scores.jsonl --n 20 --noise-filter \\
        --output aa_lcr_subset.json

Step 2 — evaluate a new model::

    evalscope eval --model <model> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"subset_indices": [3, 4, 68, ...]}'

Notes
-----
* ``subset_indices`` are 0-based positions in the full 100-item AA-LCR dataset.
* The noise filter is on by default for inline pruning (recommended because
  AA-LCR uses an LLM judge which introduces non-deterministic score variance).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter, PROMPT_TEMPLATE
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

SDS-pruned subset of AA-LCR. At 20 items (20% of the full benchmark) the
per-model accuracy estimates stay within ±4% of the full-set result.  The
LLM-judge noise filter suppresses non-deterministic scoring artefacts.

## Usage

Pass ``subset_indices`` (from ``evalscope prune --noise-filter``) **or**
set ``prune_ratio`` + ``scores_path`` to run inline SDS on startup.
""",
        dataset_id='evalscope/AA-LCR',
        subset_list=['default'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local AA-LCR text directory; auto-downloaded if null.',
                'value': None,
            },
            'subset_indices': {
                'type': 'list[int] | null',
                'description': 'Pre-computed item positions (from ``evalscope prune``).',
                'value': None,
            },
            'prune_ratio': {
                'type': 'float | null',
                'description': 'Fraction of items to select via SDS (e.g. 0.20). Requires scores_path.',
                'value': None,
            },
            'scores_path': {
                'type': 'str | null',
                'description': 'Path to JSONL score matrix for inline SDS.',
                'value': None,
            },
            'pruning_strategy': {
                'type': 'str',
                'description': 'Pruning strategy. Only "sds" is supported.',
                'value': 'sds',
            },
            'noise_filter': {
                'type': 'bool',
                'description': 'Apply LLM-judge noise filter before SDS (default True for AA-LCR).',
                'value': True,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """AA-LCR adapter restricted to an SDS-selected item subset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._allowed_positions: set[int] | None = self._resolve_indices()
        self._record_counter: int = 0

        n_desc = len(self._allowed_positions) if self._allowed_positions is not None else 'all'
        logger.info(f'[aa_lcr_pruned] Subset: {n_desc} items.')

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------

    def _resolve_indices(self) -> set[int] | None:
        """Return the set of allowed record positions, or None to keep all."""
        raw = self.extra_params.get('subset_indices')
        if raw:
            return {int(i) for i in raw}

        prune_ratio = self.extra_params.get('prune_ratio')
        scores_path = self.extra_params.get('scores_path')
        if prune_ratio is not None and scores_path:
            return self._run_sds(str(scores_path), float(prune_ratio))

        return None

    def _run_sds(self, scores_path: str, prune_ratio: float) -> set[int]:
        """Load a score matrix JSONL and select items via SDS."""
        import numpy as np
        from evalscope.pruners import SDSPruner, estimate_noise_weights

        rows = [json.loads(line) for line in Path(scores_path).open() if line.strip()]
        models = sorted(rows[0]['scores'].keys())
        score_matrix = np.array(
            [[row['scores'][m] for m in models] for row in rows], dtype=float
        )
        metadata = [row.get('metadata') or {} for row in rows]

        # Noise filter on by default for judge-scored benchmarks.
        if self.extra_params.get('noise_filter', True):
            weights = estimate_noise_weights(score_matrix)
            score_matrix = score_matrix * weights[:, None]

        n = max(1, round(len(rows) * prune_ratio))
        indices = SDSPruner().prune(score_matrix, n=n, metadata=metadata)
        logger.info(
            f'[aa_lcr_pruned] SDS: {len(indices)}/{len(rows)} items '
            f'({100 * len(indices) / len(rows):.0f}%).'
        )
        return set(indices)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        sample.metadata['_prune_position'] = self._record_counter
        self._record_counter += 1
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        if self._allowed_positions is None:
            return True
        return sample.metadata.get('_prune_position') in self._allowed_positions
