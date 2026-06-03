# Copyright (c) Alibaba, Inc. and its affiliates.
"""
LiveCodeBench Pruned Adapter.

Registers ``live_code_bench_pruned`` — an SDS-selected subset of LCB v5.

Quick start
-----------
Step 1 — build the subset once (offline, from existing model scores)::

    evalscope prune --scores lcb_scores.jsonl --n 35 --output lcb_subset.json

Step 2 — evaluate a new model on only those items::

    evalscope eval --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"subset_indices": [0, 1, 5, ...]}'

Or run SDS selection inline::

    evalscope eval --model <model> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"prune_ratio": 0.11, "scores_path": "lcb_scores.jsonl"}'

Notes
-----
* ``subset_indices`` are 0-based positions in the **unfiltered** v5 dataset, as
  produced by ``evalscope prune``.
* Date filtering is disabled — the pruned subset is already difficulty-calibrated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

SDS-pruned subset of LiveCodeBench v5.  Evaluating only the selected items
gives per-model accuracy estimates within ±1% of the full benchmark at ≈11%
of the compute cost.

## Usage

Pass ``subset_indices`` (a pre-computed list from ``evalscope prune``) **or**
set ``prune_ratio`` + ``scores_path`` to run SDS selection inline.

## Notes

* Date filtering is disabled; the pruned subset is already difficulty-calibrated.
* ``subset_indices`` are 0-based positions in the unfiltered LCB v5 dataset.
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n{format_prompt} '
            '### Answer: (use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
        extra_params={
            'subset_indices': {
                'type': 'list[int] | null',
                'description': 'Pre-computed item positions (from ``evalscope prune``).',
                'value': None,
            },
            'prune_ratio': {
                'type': 'float | null',
                'description': 'Fraction of items to select via SDS (e.g. 0.11). Requires scores_path.',
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
                'description': 'Apply LLM-judge noise filter before SDS.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {'shell_executor': {}, 'python_executor': {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench restricted to an SDS-selected item subset."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Disable date filtering — the pruned subset is already calibrated.
        self.start_date = None
        self.end_date = None

        self._allowed_positions: set[int] | None = self._resolve_indices()
        self._record_counter: int = 0

        n_desc = len(self._allowed_positions) if self._allowed_positions is not None else 'all'
        logger.info(f'[live_code_bench_pruned] Subset: {n_desc} items.')

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

        if self.extra_params.get('noise_filter'):
            weights = estimate_noise_weights(score_matrix)
            score_matrix = score_matrix * weights[:, None]

        n = max(1, round(len(rows) * prune_ratio))
        indices = SDSPruner().prune(score_matrix, n=n, metadata=metadata)
        logger.info(
            f'[live_code_bench_pruned] SDS: {len(indices)}/{len(rows)} items '
            f'({100 * len(indices) / len(rows):.0f}%).'
        )
        return set(indices)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        # Tag with raw dataset position (before any date filtering).
        sample.metadata['_prune_position'] = self._record_counter
        self._record_counter += 1
        return sample

    def sample_filter(self, sample: Sample) -> bool:
        # super().sample_filter() applies date filtering; with start_date=None
        # and end_date=None it always returns True.  Call it to stay correct if
        # the parent adds additional filter logic in the future.
        if not super().sample_filter(sample):
            return False
        if self._allowed_positions is None:
            return True
        return sample.metadata.get('_prune_position') in self._allowed_positions
