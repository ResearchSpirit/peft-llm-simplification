---
base_model: unsloth/gemma-3-1b-it-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: gemma
language:
- en
datasets:
- cbasu/Med-EASi
tags:
- lora
- qlora
- text-simplification
- sentence-simplification
- unsloth
---

# Gemma-3 1B QLoRA adapter for Med-EASi sentence simplification

LoRA adapter from the ICoDiT 2026 paper *Parameter-Efficient Fine-Tuning of Sub-2B Language
Models for General and Biomedical Sentence Simplification* (Wijaya & Philip, BINUS University).
Code, predictions and evaluation: https://github.com/ResearchSpirit/peft-llm-simplification

| | |
|---|---|
| Base model | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it), loaded 4-bit as `unsloth/gemma-3-1b-it-bnb-4bit` |
| Training data | [Med-EASi](https://huggingface.co/datasets/cbasu/Med-EASi) (biomedical), 1,397 pairs from the official `train` split |
| Method | QLoRA (4-bit NF4, double quantisation), r = 16, alpha = 32, dropout 0, all attention + MLP projections |
| Optimisation | AdamW-8bit, lr 2e-4, linear schedule, 3% warm-up, 2 epochs, effective batch 8, seed 42, loss on response tokens only |
| Trainable parameters | 13.05 M (1.29% of the model) |
| Prompt format | the model's native chat template |
| Decoding used in the paper | greedy, repetition penalty 1.1, max 256 new tokens |

The exact prompt and decoding code is in `final_notebook.ipynb` (Sections 04 and 07) of the
repository above; use it to reproduce the published predictions. Results, efficiency
figures and limitations are reported in the paper and in the repository README.

**Intended use.** Research on sentence simplification. Outputs are not checked for factual
accuracy; do not use them for patient-facing or other high-stakes text without expert review.

**License.** Use of this adapter is subject to the license of the base model ([Gemma Terms of Use](https://ai.google.dev/gemma/terms))
and of the training dataset.
