# Copyright (c) Alibaba, Inc. and its affiliates.
"""
MMUM Encoder Probe Pruner (Part B).

Selects a subject-stratified probe set from MMMU that specifically stresses the
image encoder.  A model with a degraded encoder will fail these items reliably,
whereas random MMMU sampling dilutes the signal because many MMMU questions are
answerable from the question text without reading the image.

Design rationale
----------------
An encoder degrades when it loses fine visual detail, not when it fails on
questions answerable from text + world knowledge alone.  The probe maximises
the separation between "encoder broken" and "model reasoning broken" by
selecting items where the answer is **contingent on the image content**:

1. **Subject selection** — 6 subjects with high encoder dependency:
   Art_Theory, Chemistry, Clinical_Medicine, Computer_Science,
   Diagnostics_and_Laboratory_Medicine, Mechanical_Engineering.

2. **Image-type priority** — within each subject, items containing high-stress
   image types (diagrams, medical images, charts, schematics) are preferred.

3. **Topic difficulty** — harder items require more precise visual reading.

4. **Visual-reference keywords** — questions that reference the image explicitly
   ("in the figure", "shown in", "depicted above") depend on encoder output.

Usage
-----
::

    from evalscope.pruners import EncoderProbePruner

    pruner = EncoderProbePruner(items_per_subject=15)
    # records: list of MMMU record dicts (from HF/ModelScope dataset)
    selected_indices = pruner.select_from_records(records)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BasePruner


# ---------------------------------------------------------------------------
# Subject and image-type constants
# ---------------------------------------------------------------------------

HIGH_ENCODER_STRESS_SUBJECTS: frozenset[str] = frozenset({
    'Art_Theory',
    'Chemistry',
    'Clinical_Medicine',
    'Computer_Science',
    'Diagnostics_and_Laboratory_Medicine',
    'Mechanical_Engineering',
})

HIGH_STRESS_IMG_TYPES: frozenset[str] = frozenset({
    'Diagram', 'Chart', 'Medical Image', 'Scientific Image',
    'Table', 'Graph', 'Schematic', 'Chemical Structure',
    'Microscopy', 'MRI/CT Scan', 'Pathology', 'X-ray',
    'Circuit', 'Blueprint', 'Floor Plan', 'Engineering Drawing',
})

_VISUAL_REF_PATTERN = re.compile(
    r'\b(figure|shown|depicted|graph|table|image|diagram|chart|'
    r'illustrat|calculat.*above|above.*calculat|specimen|slide|'
    r'compound|structure|circuit)\b',
    re.IGNORECASE,
)

_DIFFICULTY_SCORE: Dict[str, float] = {'Easy': 0.0, 'Medium': 1.0, 'Hard': 2.0}


# ---------------------------------------------------------------------------
# Per-item encoder stress score (also used by MMMUEncoderProbeAdapter)
# ---------------------------------------------------------------------------

def _encoder_stress(record: Dict[str, Any]) -> float:
    """Compute a per-item encoder-stress score (higher = more encoder-dependent).

    Three additive components
    -------------------------
    * Image type:       +2 if the image type demands precise visual decoding.
    * Topic difficulty: +0..+2 based on MMMU annotator grade.
    * Visual keywords:  +0.5 per keyword reference, capped at +2.
    """
    score = 0.0

    img_type = record.get('img_type', '') or ''
    if any(t.lower() in img_type.lower() for t in HIGH_STRESS_IMG_TYPES):
        score += 2.0

    difficulty = record.get('topic_difficulty', 'Medium') or 'Medium'
    score += _DIFFICULTY_SCORE.get(difficulty, 1.0)

    question = record.get('question', '') or ''
    matches = _VISUAL_REF_PATTERN.findall(question)
    score += min(len(matches) * 0.5, 2.0)

    return score


# ---------------------------------------------------------------------------
# EncoderProbePruner
# ---------------------------------------------------------------------------

class EncoderProbePruner(BasePruner):
    """Subject-stratified MMMU probe pruner targeting image encoder degradation.

    Unlike :class:`SDSPruner`, which requires a multi-model score matrix, this
    pruner works from raw dataset records and ranks items by structural
    properties — no model inference required.

    Parameters
    ----------
    items_per_subject:
        Items to select from each high-encoder-stress subject.
        Default 15 → 6 subjects × 15 = 90 items ≈ 0.75% of the 12K MMMU dataset.
    subjects:
        Override the set of high-stress subjects.  Defaults to the 6 subjects
        identified in the analysis (see module docstring).
    """

    def __init__(
        self,
        items_per_subject: int = 15,
        subjects: Optional[frozenset] = None,
    ) -> None:
        self.items_per_subject = items_per_subject
        self.subjects = subjects if subjects is not None else HIGH_ENCODER_STRESS_SUBJECTS

    def prune(
        self,
        score_matrix: np.ndarray,
        n: int,
        *,
        metadata: Optional[Sequence[dict]] = None,
    ) -> List[int]:
        """BasePruner interface.  Delegates to ``select_from_records``.

        For MMMU, pass raw records as ``metadata``; ``score_matrix`` and ``n``
        are ignored because selection is driven by structural item properties.
        """
        if metadata is None:
            raise ValueError(
                'EncoderProbePruner requires metadata=<list of raw MMMU records>.'
            )
        return self.select_from_records(list(metadata))

    def select_from_records(self, records: List[Dict[str, Any]]) -> List[int]:
        """Select encoder-stress items from a flat list of MMMU records.

        Parameters
        ----------
        records:
            List of raw MMMU records, each containing at minimum
            ``question``, ``img_type``, ``topic_difficulty``.

        Returns
        -------
        list[int]
            Sorted row indices into ``records``.
        """
        scores = np.array([_encoder_stress(r) for r in records])
        ranked = np.argsort(-scores, kind='stable')
        take = min(self.items_per_subject, len(ranked))
        return sorted(int(i) for i in ranked[:take])

    def select_subject_subset(
        self,
        all_records: List[Dict[str, Any]],
        subject_field: str = 'subfield',
    ) -> Dict[str, List[int]]:
        """Select items per subject from a flat multi-subject record list.

        Parameters
        ----------
        all_records:
            Flat list of MMMU records across all subjects.
        subject_field:
            Record field identifying the subject (e.g. ``'subfield'``).

        Returns
        -------
        Dict[str, List[int]]
            Mapping from subject name → sorted global record indices.
        """
        by_subject: Dict[str, List[int]] = defaultdict(list)
        for global_idx, record in enumerate(all_records):
            subject = record.get(subject_field, '')
            if subject in self.subjects:
                by_subject[subject].append(global_idx)

        result: Dict[str, List[int]] = {}
        for subject, indices in by_subject.items():
            subject_records = [all_records[i] for i in indices]
            local_selected = self.select_from_records(subject_records)
            result[subject] = sorted(indices[j] for j in local_selected)

        return result

    def __repr__(self) -> str:
        return (
            f'EncoderProbePruner('
            f'items_per_subject={self.items_per_subject}, '
            f'subjects={len(self.subjects)} subjects)'
        )
