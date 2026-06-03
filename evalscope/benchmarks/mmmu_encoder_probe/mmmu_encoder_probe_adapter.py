# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MMUM Encoder Probe Adapter (Part B).

Registers ``mmmu_encoder_probe`` — a subject-stratified subset of MMMU that
targets image encoder degradation.

Algorithm
---------
Six subjects with high encoder dependency are loaded.  Within each subject,
items are ranked by an encoder-stress score (image type × topic difficulty ×
visual-reference keyword density in the question text) and the top
``items_per_subject`` are kept.

Default probe: 6 subjects × 15 items = **90 items** ≈ 0.75% of the full 12K MMMU.

A model with a degraded encoder will fail these items reliably.  Random MMMU
sampling dilutes the signal because many MMMU questions are answerable from
the question text alone, without reading the image.

Quick start
-----------
::

    evalscope eval --model <model> \\
        --datasets mmmu_encoder_probe \\
        --dataset-args '{"items_per_subject": 15}'
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, MemoryDataset, Sample
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope.constants import Tags
from evalscope.pruners.encoder_probe_pruner import HIGH_ENCODER_STRESS_SUBJECTS, _encoder_stress
from evalscope.utils.logger import get_logger

logger = get_logger()

_STRESS_SUBJECTS: List[str] = sorted(HIGH_ENCODER_STRESS_SUBJECTS)


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_encoder_probe',
        pretty_name='MMMU (Encoder Probe)',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description="""
## Overview

Encoder-stress probe derived from MMMU.  Selects items from 6 subjects where
the correct answer requires reading specific image content (charts, medical
images, engineering diagrams).  A model with a degraded image encoder will
fail these items reliably before the degradation appears in overall MMMU
accuracy.

## Subjects

Art_Theory, Chemistry, Clinical_Medicine, Computer_Science,
Diagnostics_and_Laboratory_Medicine, Mechanical_Engineering.

## Probe size

Default: 6 subjects × 15 items = **90 items** ≈ 0.75% of the full 12K MMMU.
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=_STRESS_SUBJECTS,
        metric_list=['acc'],
        eval_split='validation',
        extra_params={
            'items_per_subject': {
                'type': 'int',
                'description': 'Items to keep per high-stress subject (default 15).',
                'value': 15,
            },
        },
    )
)
class MMMUEncoderProbeAdapter(MMMUAdapter):
    """MMMU adapter restricted to high-encoder-stress items."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._items_per_subject: int = int(
            self.extra_params.get('items_per_subject') or 15
        )

    # ------------------------------------------------------------------
    # Loading — post-process each subject to top-N encoder-stress items
    # ------------------------------------------------------------------

    def load(self):
        test_dataset, fewshot_dataset = super().load()
        filtered = self._apply_encoder_probe_filter(test_dataset)
        return filtered, fewshot_dataset

    def _apply_encoder_probe_filter(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Keep only the top ``_items_per_subject`` encoder-stress items per subject."""
        total = 0
        for subject_name in list(dataset_dict.keys()):
            dataset = dataset_dict[subject_name]
            samples: List[Sample] = list(dataset)
            if not samples:
                continue

            stress_scores = np.array([
                _encoder_stress({
                    'img_type': s.metadata.get('img_type', ''),
                    'topic_difficulty': s.metadata.get('topic_difficulty', 'Medium'),
                    'question': s.metadata.get('_raw_question', ''),
                })
                for s in samples
            ])

            ranked = np.argsort(-stress_scores, kind='stable')
            take = min(self._items_per_subject, len(ranked))
            keep_set = set(int(i) for i in ranked[:take])

            kept = [s for i, s in enumerate(samples) if i in keep_set]
            for new_i, s in enumerate(kept):
                s.id = new_i
                s.group_id = new_i

            dataset_dict[subject_name] = MemoryDataset(
                samples=kept,
                name=getattr(dataset, '_name', subject_name),
                location=getattr(dataset, '_location', None),
            )
            total += len(kept)
            logger.info(
                f'[mmmu_encoder_probe] {subject_name}: '
                f'{len(kept)}/{len(samples)} items selected.'
            )

        logger.info(f'[mmmu_encoder_probe] Total probe size: {total} items.')
        return dataset_dict

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        # Store raw question text for encoder-stress ranking.
        # img_type and topic_difficulty are already in metadata from MMMUAdapter.
        sample.metadata['_raw_question'] = record.get('question', '')
        return sample
