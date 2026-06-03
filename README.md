> **This is a fork of [modelscope/evalscope](https://github.com/modelscope/evalscope).**
>
> **Pinned upstream SHA**: `de7b0b3f08c617f48a00ef09f7169dc74212a6d9`
>
> **What was added** (AI Model Quality Challenge — Task 2):
>
> | Path | What it is |
> |---|---|
> | `evalscope/pruners/base.py` | `BasePruner` abstract class |
> | `evalscope/pruners/sds_pruner.py` | Stratified Discrimination Sampling (SDS) |
> | `evalscope/pruners/noise_filter.py` | LLM-judge noise filter for AA-LCR |
> | `evalscope/pruners/encoder_probe_pruner.py` | MMMU encoder-stress pruner (Part B) |
> | `evalscope/benchmarks/live_code_bench_pruned/` | Registered dataset `live_code_bench_pruned` |
> | `evalscope/benchmarks/aa_lcr_pruned/` | Registered dataset `aa_lcr_pruned` |
> | `evalscope/benchmarks/mmmu_encoder_probe/` | Registered dataset `mmmu_encoder_probe` (Part B) |
> | `evalscope/cli/start_prune.py` | `evalscope prune` CLI command |
> | `evalscope/tools/compare_runs.py` | `python -m evalscope.tools.compare_runs` |
>
> **Install**:
> ```bash
> pip install git+https://github.com/andrey-savov/evalscope
> ```
>
> **Run the pruner** (Part A):
> ```bash
> evalscope prune --scores lcb_scores.jsonl --n 35 --output lcb_subset.json
> evalscope eval --model <model> --datasets live_code_bench_pruned \
>     --dataset-args '{"prune_ratio": 0.11, "scores_path": "lcb_scores.jsonl"}'
> python -m evalscope.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/
> ```
>
> **Run the MMMU encoder probe** (Part B):
> ```bash
> evalscope eval --model <model> --datasets mmmu_encoder_probe \
>     --dataset-args '{"items_per_subject": 15}'
> ```

---

<p align="center">
    <br>
    <img src="docs/en/_static/images/evalscope_logo.png"/>
    <br>
<p>

<p align="center">
  <a href="README_zh.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
<a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope"></a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<a href='https://evalscope.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' /></a>
<p>

<p align="center">
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a> &nbsp ｜ &nbsp <a href="https://evalscope.readthedocs.io/en/latest/"> 📖  English Documentation</a>
<p>


> ⭐ If you like this project, please click the "Star" button in the upper right corner to support us. Your support is our motivation to move forward!

## 📝 Introduction

EvalScope is a one-stop LLM evaluation framework built by the [ModelScope Community](https://modelscope.cn/). Just one command to start — it supports model capability evaluation, inference performance stress testing, and result visualization.

```bash
pip install evalscope
evalscope eval --model your-model-name --api-url $OPENAI_API_BASE_URL --api-key $OPENAI_API_KEY --eval-type openai_api --datasets gsm8k --limit 5
```

## ✨ Key Features

- **📚 Comprehensive Evaluation Benchmarks**: Built-in multiple industry-recognized evaluation benchmarks including MMLU, C-Eval, GSM8K, and more.
- **🧩 Multi-modal and Multi-domain Support**: Supports evaluation of various model types including Large Language Models (LLM), Vision Language Models (VLM), Embedding, Reranker, AIGC, and more.
- **🚀 Multi-backend Integration**: Seamlessly integrates multiple evaluation backends including OpenCompass, VLMEvalKit, RAGEval to meet different evaluation needs.
- **🤖 Agent Evaluation Mode**: Drives benchmarks (e.g. GSM8K, AIME, SWE-bench Agentic) inside a controlled multi-turn AgentLoop with pluggable strategies, tools and Docker sandbox; full per-sample Agent Trace is recorded and visualizable.
- **⚡ Inference Performance Testing**: Provides powerful model service stress testing tools, supporting multiple performance metrics such as TTFT, TPOT.
- **📊 Interactive Reports**: Provides WebUI visualization interface, supporting multi-dimensional model comparison, report overview and detailed inspection.
- **⚔️ Arena Mode**: Supports multi-model battles (Pairwise Battle), intuitively ranking and evaluating models.
- **🔧 Highly Extensible**: Developers can easily add custom datasets, models and evaluation metrics.
