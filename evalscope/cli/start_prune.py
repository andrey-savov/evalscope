# Copyright (c) Alibaba, Inc. and its affiliates.
"""
CLI command: ``evalscope prune``

Reads a pre-computed score matrix (JSONL format), runs the SDS pruner, and
writes the selected item indices to a JSON file.

Score matrix JSONL format
-------------------------
One JSON object per line::

    {"index": 0, "scores": {"model_a": 1.0, "model_b": 0.0, "model_c": 1.0},
     "metadata": {"input_tokens": 1234}}
    {"index": 1, "scores": {"model_a": 0.0, "model_b": 0.0, "model_c": 0.0}, ...}
    ...

Example
-------
::

    evalscope prune \\
        --scores lcb_scores.jsonl \\
        --n 35 \\
        --output lcb_selected.json

The output JSON is a list of item indices that can be passed to the evalscope
dataset loader via the ``subset_indices`` argument::

    task_cfg = TaskConfig(
        ...,
        dataset_args={"subset_indices": selected_indices},
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _subparser_func(args: argparse.Namespace) -> "PruneCMD":
    return PruneCMD(args)


class PruneCMD:
    name = "prune"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @staticmethod
    def define_args(parsers: argparse._SubParsersAction) -> None:
        parser = parsers.add_parser(
            PruneCMD.name,
            help="Prune a benchmark to a representative subset using SDS.",
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--scores",
            required=True,
            metavar="PATH",
            help="Path to JSONL file with pre-computed score matrix.",
        )
        parser.add_argument(
            "--n",
            required=True,
            type=int,
            metavar="N",
            help="Target subset size.",
        )
        parser.add_argument(
            "--output",
            default="selected_indices.json",
            metavar="PATH",
            help="Output JSON file for selected item indices (default: selected_indices.json).",
        )
        parser.add_argument(
            "--n-bands",
            type=int,
            default=10,
            metavar="K",
            help="Number of difficulty bands for SDS (default: 10).",
        )
        parser.add_argument(
            "--noise-filter",
            action="store_true",
            default=False,
            help="Apply LLM-judge noise filter before pruning (recommended for judge-scored benchmarks).",
        )
        parser.set_defaults(func=_subparser_func)

    def execute(self) -> None:
        import numpy as np

        from evalscope.pruners import SDSPruner, estimate_noise_weights

        scores_path = Path(self.args.scores)
        if not scores_path.exists():
            print(f"[prune] Error: scores file not found: {scores_path}", file=sys.stderr)
            sys.exit(1)

        # ── Load score matrix ──────────────────────────────────────────────
        rows = []
        with scores_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            print("[prune] Error: scores file is empty.", file=sys.stderr)
            sys.exit(1)

        # Determine model order from first row
        first_scores = rows[0]["scores"]
        models = sorted(first_scores.keys())
        n_items = len(rows)

        score_matrix = np.zeros((n_items, len(models)), dtype=float)
        metadata: list[dict] = []
        for i, row in enumerate(rows):
            for j, m in enumerate(models):
                score_matrix[i, j] = float(row["scores"].get(m, 0.0))
            metadata.append(row.get("metadata") or {})

        print(f"[prune] Loaded {n_items} items x {len(models)} models from {scores_path.name}")
        print(f"[prune] Models: {models}")
        print(f"[prune] Mean accuracy: { {m: round(float(score_matrix[:, j].mean()), 3) for j, m in enumerate(models)} }")

        # ── Optional noise filter ──────────────────────────────────────────
        if self.args.noise_filter:
            weights = estimate_noise_weights(score_matrix)
            n_noisy = int((weights < 1.0).sum())
            print(f"[prune] Noise filter: {n_noisy} potentially noisy items down-weighted.")
            score_matrix = score_matrix * weights[:, None]

        # ── Run SDS pruner ─────────────────────────────────────────────────
        pruner = SDSPruner(n_bands=self.args.n_bands)
        indices = pruner.prune(score_matrix, n=self.args.n, metadata=metadata)

        # ── Report ─────────────────────────────────────────────────────────
        sub_acc = score_matrix[indices, :].mean(axis=0)
        full_acc = score_matrix.mean(axis=0)
        mae = float(np.abs(sub_acc - full_acc).mean())

        print(f"[prune] Selected {len(indices)} / {n_items} items ({100*len(indices)/n_items:.0f}%)")
        print(f"[prune] Subset vs full-set accuracy MAE: {mae:.4f}")
        for j, m in enumerate(models):
            print(f"[prune]   {m}: full={full_acc[j]:.3f}  subset={sub_acc[j]:.3f}  err={sub_acc[j]-full_acc[j]:+.3f}")

        # ── Write output ───────────────────────────────────────────────────
        out_path = Path(self.args.output)
        out_path.write_text(json.dumps(indices, indent=2))
        print(f"[prune] Indices written to {out_path}")
