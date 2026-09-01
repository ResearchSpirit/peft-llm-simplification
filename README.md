# Scaling Down Simplicity: Parameter-Efficient Fine-Tuning of Sub-2B Language Models for General and Biomedical Text Simplification

Research repository for the paper submitted to the **1st International Conference on Computing and Digital Technology (ICoDiT) 2026**, track: Intelligent Systems & Artificial Intelligence (ISAI).

**Authors:** Stanley Nathanael Wijaya, Samuel Philip — School of Computer Science, Bina Nusantara University, Jakarta, Indonesia

---

## Overview

This study investigates whether open-weight language models with under two billion parameters can achieve competitive sentence simplification through supervised parameter-efficient fine-tuning, without relying on large-scale compute or closed APIs.

Three sub-2B models are fine-tuned using Low-Rank Adaptation (LoRA) and 4-bit NF4 quantization on two benchmarks — the general-domain ASSET dataset and the biomedical Med-EASi dataset — and evaluated against zero-shot baselines and previously reported large LLM results.

**Key finding:** Fine-tuning effectiveness is model-dependent and domain-dependent. Gemma-3 1B achieves the strongest SARI gains on both datasets; Qwen3 1.7B demonstrates that fine-tuning does not universally improve simplification quality; DeepSeek-R1-Distill-Qwen 1.5B provides the fastest fine-tuned inference while also gaining on ASSET.

---

## Models

| Model | HuggingFace ID | Parameters | Trainable (LoRA) | Adapter Size |
|---|---|---|---|---|
| Gemma-3 1B | `google/gemma-3-1b-it` | 1,012.9 M | 13.05 M (1.3%) | 52.2 MB |
| Qwen3 1.7B | `Qwen/Qwen3-1.7B` | 1,738.0 M | 17.43 M (1.0%) | 69.8 MB |
| DeepSeek-R1-Distill-Qwen 1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1,795.6 M | 18.46 M (1.0%) | 73.9 MB |

All models are loaded via [Unsloth](https://github.com/unslothai/unsloth) 4-bit BnB checkpoints.

---

## Datasets

| Dataset | HuggingFace ID | Domain | Train Split Used | Test Split Used |
|---|---|---|---|---|
| ASSET | `facebook/asset` | General (Wikipedia) | `validation` — 2,000 pairs | `test` — 359 sentences |
| Med-EASi | `cbasu/Med-EASi` | Biomedical | `train` — 1,397 pairs | `test` — 300 sentences |

ASSET provides 10 crowdsourced simplified references per source sentence. Med-EASi provides a single reference per pair. The Med-EASi evaluation includes a sensitivity analysis on 282 non-overlapping instances (18 of 300 test sources overlap with training).

---

## Results

### ASSET (General Domain)

| Model | Setting | SARI ↑ | BERTScore (raw) ↑ | BERTScore (rescaled) ↑ | FKGL ↓ |
|---|---|---|---|---|---|
| Copy source (identity) | — | 53.78 | 98.63 | 91.91 | 11.64 |
| Gold reference (LOO) | — | 43.47 | 96.59 | 79.79 | 8.32 |
| Gemma-3 1B | Zero-shot | 40.52 | 83.99 | 5.12 | 11.91 |
| Qwen3 1.7B | Zero-shot | 44.11 | 96.57 | 79.69 | 9.37 |
| DeepSeek-Distill 1.5B | Zero-shot | 43.19 | 92.08 | 53.05 | 11.18 |
| **Gemma-3 1B** | **Fine-tuned** | **46.57** | **97.54** | **85.45** | **8.75** |
| Qwen3 1.7B | Fine-tuned | 41.96 | 96.31 | 78.11 | 7.94 |
| DeepSeek-Distill 1.5B | Fine-tuned | 44.71 | 97.02 | 82.32 | 8.50 |

*Literature baselines (reported by Qiang et al., 2025; not re-run here): MUSS SARI 42.29, LLaMA3.1 70B SARI 47.27, GPT-4o SARI 49.20.*

### Med-EASi (Biomedical Domain)

| Model | Setting | SARI ↑ | BERTScore (raw) ↑ | BERTScore (rescaled) ↑ | FKGL ↓ |
|---|---|---|---|---|---|
| Copy source (identity) | — | 49.24 | 92.04 | 52.86 | 13.64 |
| Gemma-3 1B | Zero-shot | 37.93 | 82.26 | −5.10 | 12.38 |
| Qwen3 1.7B | Zero-shot | 38.36 | 90.78 | 45.37 | 10.27 |
| DeepSeek-Distill 1.5B | Zero-shot | 37.83 | 87.68 | 27.03 | 12.62 |
| **Gemma-3 1B** | **Fine-tuned** | **40.27** | **90.62** | **44.44** | **10.62** |
| Qwen3 1.7B | Fine-tuned | 37.36 | 88.89 | 34.18 | 10.54 |
| DeepSeek-Distill 1.5B | Fine-tuned | 37.09 | 89.52 | 37.92 | 9.89 |

*Literature baselines (reported by Qiang et al., 2025; not re-run here): MUSS SARI 35.15, LLaMA3.1 70B SARI 39.55, GPT-4o SARI 40.81.*

### Statistical Significance (Paired Bootstrap, Fine-tuned vs. Zero-shot)

| Model | Dataset | ΔSARI | 95% CI | Significant |
|---|---|---|---|---|
| Gemma-3 1B | ASSET | +6.05 | [5.04, 7.09] | ✅ Yes |
| DeepSeek-Distill 1.5B | ASSET | +1.53 | [0.29, 2.82] | ✅ Yes |
| Qwen3 1.7B | ASSET | −2.15 | — | ❌ No (fine-tuning reduces SARI) |
| Gemma-3 1B | Med-EASi (n=300) | +2.35 | — | ✅ Yes |
| Gemma-3 1B | Med-EASi (n=282, non-overlapping) | +1.25 | [−0.56, 2.89] | ❌ Not significant (p=0.097) |

---

## Fine-Tuning Configuration

| Hyperparameter | Value |
|---|---|
| Method | QLoRA (4-bit NF4, double quantization) |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0.0 |
| Target layers | `q/k/v/o_proj`, `gate/up/down_proj` |
| Epochs | 2 |
| Learning rate | 2×10⁻⁴ |
| Optimizer | AdamW 8-bit |
| LR scheduler | Linear (warmup ratio 0.03) |
| Effective batch size | 8 (per-device 2, grad. accum. 4) |
| Max sequence length | 512 |
| Seed | 42 |
| Loss | Response-only (prompt tokens masked) |
| Qwen3 thinking mode | Disabled (consistently in training and inference) |

**Inference:** Greedy decoding, repetition penalty 1.1, max 256 new tokens.

---

## Efficiency

| Model | Train Time (ASSET) | Fine-tuned Latency | Peak GPU Memory | Adapter Size |
|---|---|---|---|---|
| Gemma-3 1B | ~20.6 min | ~3.59 s/sample | 1.67 GB | 52.2 MB |
| Qwen3 1.7B | ~15.5 min | ~1.72 s/sample | 3.15 GB | 69.8 MB |
| DeepSeek-Distill 1.5B | ~16.5 min | ~1.60 s/sample | 2.29 GB | 73.9 MB |

All experiments were run on a single NVIDIA Tesla T4 (15.64 GB VRAM) via Google Colab.

---

## Repository Structure

```
.
├── final_notebook.ipynb          # Main experiment pipeline (single source of truth)
├── outputs/
│   ├── configs/                  # Experiment, training, and inference configuration JSONs
│   │   ├── experiment_config.json
│   │   ├── training_config.json
│   │   ├── training_stats.json
│   │   ├── inference_stats.json
│   │   ├── dataset_stats.json
│   │   ├── evaluation_subset.json
│   │   ├── run_provenance.json
│   │   └── environment.json
│   ├── figures/                  # All generated figures (PNG)
│   ├── metrics/                  # Evaluation metrics (CSV)
│   │   ├── metrics_all_systems.csv
│   │   ├── bootstrap_confidence_intervals.csv
│   │   ├── significance_ft_vs_zeroshot_clean.csv
│   │   ├── efficiency_metrics.csv
│   │   ├── editing_behaviour.csv
│   │   ├── per_sentence_scores_asset.csv
│   │   ├── per_sentence_scores_medeasi.csv
│   │   ├── qualitative_examples.csv
│   │   ├── decoding_artefacts.csv
│   │   ├── leakage_sensitivity.csv
│   │   └── sari_sanity_probes.csv
│   ├── predictions/              # Raw and processed model outputs (CSV)
│   │   ├── asset_predictions.csv
│   │   ├── medeasi_predictions.csv
│   │   ├── ft_<model>_<dataset>_raw.csv
│   │   └── zeroshot_<model>_<dataset>_raw.csv
│   ├── tables/                   # LaTeX and CSV result tables
│   └── requirements.txt          # Pinned dependency list
├── workspace/
│   ├── paper.tex                 # Final submitted paper (IEEE format, ICoDiT)
│   ├── main.tex                  # Springer LLNCS format manuscript
│   ├── template.tex              # Springer LLNCS template
│   ├── references.bib            # BibTeX references
│   ├── claim_audit.md            # Claim-to-evidence traceability table
│   ├── paper_checklist.md        # Submission compliance checklist
│   ├── REVISION_SUMMARY.md       # Summary of revisions between drafts
│   ├── rev.md / rev-final.md     # Revision notes
│   ├── audit_numbers.py          # Script to cross-check numbers in paper vs. outputs
│   └── check_tex.py              # LaTeX consistency checker
├── prompts/
│   ├── build.md                  # Master prompt used for AI-assisted manuscript preparation
│   ├── claude.md                 # Claude-specific prompting notes
│   └── requirements.txt          # Additional prompt-related dependencies
├── guidelines/                   # ICoDiT conference author guidelines (PDF)
├── turnitin/                     # Plagiarism and AI-detection reports
└── .gitignore
```

---

## Reproducing the Experiment

The full pipeline is self-contained in [`final_notebook.ipynb`](final_notebook.ipynb). A single **Restart → Run All** reproduces every metric, figure, and table from scratch in approximately 7–9 hours on a T4 GPU.

To validate the pipeline end-to-end before a full run, set `SMOKE_TEST = True` in Section 01 of the notebook (~10 minutes).

### Environment

| Dependency | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.5.0 |
| PEFT | 0.20.0 |
| Datasets | 4.3.0 |
| Evaluate | 0.4.6 |
| bert-score | 0.3.12 |
| Unsloth | 2026.8.22 |
| NumPy | 2.1.3 |
| Pandas | 2.2.3 |
| Matplotlib | 3.10.0 |

Full pinned dependency list: [`outputs/requirements.txt`](outputs/requirements.txt)

### Quick Setup (Google Colab)

Section 00 of the notebook handles all package installation — no kernel restart required afterwards. The notebook writes all outputs to a configurable `BASE_OUTPUT` path; no hardcoded paths exist.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **SARI** | Simplification-specific metric measuring edit quality relative to both source and references. Higher is better. Computed via HuggingFace `evaluate`. |
| **BERTScore F1 (raw)** | Soft token-level semantic similarity using `roberta-large` (layer 17), multi-reference (max over references). Higher is better. |
| **BERTScore F1 (rescaled)** | Same, with `rescale_with_baseline=True`. Provides a score anchored to a human paraphrase baseline. |
| **FKGL** | Flesch-Kincaid Grade Level — surface readability proxy based on sentence length and syllable count. Lower indicates simpler text. Computed via `textstat`. |

Bootstrap 95% confidence intervals and paired bootstrap significance tests (fine-tuned vs. zero-shot) are computed for SARI and FKGL across all systems.

---

## Notebook Pipeline Overview (V2)

The notebook is structured into the following sequential sections:

| Section | Description |
|---|---|
| 00 | Package installation |
| 01 | Configuration (smoke test flag, seeds, paths) |
| 02 | Dataset loading and preparation |
| 03 | Model loading (4-bit NF4 via Unsloth) |
| 04 | LoRA configuration and EOS token setup |
| 05 | Response-only loss masking (with verification) |
| 06 | Fine-tuning (6 adapters: 3 models × 2 datasets) |
| 07 | Zero-shot inference |
| 08 | Fine-tuned inference |
| 09 | SARI, BERTScore, FKGL evaluation |
| 10 | Bootstrap confidence intervals and significance tests |
| 11 | Leakage sensitivity analysis (Med-EASi n=282) |
| 12 | Editing behavior analysis (copy rate, token retention, divergence) |
| 13 | Efficiency metrics (training time, latency, GPU memory, adapter size) |
| 14 | Figures and tables |
| 15 | Provenance and configuration logging |

V2 was a full audit-driven rebuild of the original V1 pipeline, addressing 19 identified reproducibility problems. See the notebook header for a complete change log.

---

## Paper

The manuscript is prepared in two formats:
- [`workspace/paper.tex`](workspace/paper.tex) — IEEE format (for ICoDiT submission)
- [`workspace/main.tex`](workspace/main.tex) — Springer LLNCS format

**Conference:** ICoDiT 2026 — 1st International Conference on Computing and Digital Technology  
**Track:** Intelligent Systems & Artificial Intelligence (ISAI)  
**Proceedings target:** Springer Nature CCIS  
**Submission status:** Under review

---

## Citation

If you use this codebase or results, please cite the associated paper (citation will be updated upon acceptance):

```bibtex
@inproceedings{wijaya2026scalingdown,
  title     = {Fine-Tuned Lightweight Language Models for Efficient Sentence Simplification Across General and Medical Domains},
  author    = {Wijaya, Stanley Nathanael and Philip, Samuel},
  booktitle = {Proceedings of the 1st International Conference on Computing and Digital Technology (ICoDiT)},
  year      = {2026},
  publisher = {Springer Nature}
}
```

---

## License

This repository is for academic research purposes. Model weights are subject to their respective HuggingFace model licenses (Gemma, Qwen3, DeepSeek). Datasets are subject to their original licenses (ASSET, Med-EASi).
