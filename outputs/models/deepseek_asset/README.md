---
base_model: unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: mit
language:
- en
datasets:
- facebook/asset
tags:
- lora
- qlora
- text-simplification
- sentence-simplification
- unsloth
---

# DeepSeek-R1-Distill-Qwen 1.5B QLoRA adapter for ASSET sentence simplification

LoRA adapter from the ICoDiT 2026 paper *Parameter-Efficient Fine-Tuning of Sub-2B Language
Models for General and Biomedical Sentence Simplification* (Wijaya & Philip, BINUS University).
Code, predictions and evaluation: https://github.com/ResearchSpirit/peft-llm-simplification

| | |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), loaded 4-bit as `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit` |
| Training data | [ASSET](https://huggingface.co/datasets/facebook/asset) (general-domain (Wikipedia)), 2,000 pairs from the official `validation` split (first of the 10 references per source) |
| Method | QLoRA (4-bit NF4, double quantisation), r = 16, alpha = 32, dropout 0, all attention + MLP projections |
| Optimisation | AdamW-8bit, lr 2e-4, linear schedule, 3% warm-up, 2 epochs, effective batch 8, seed 42, loss on response tokens only |
| Trainable parameters | 18.46 M (1.03% of the model) |
| Prompt format | Alpaca-style instruction / input / response prompt |
| Decoding used in the paper | greedy, repetition penalty 1.1, max 256 new tokens |

The exact prompt and decoding code is in `final_notebook.ipynb` (Sections 04 and 07) of the
repository above; use it to reproduce the published predictions. Results, efficiency
figures and limitations are reported in the paper and in the repository README.

**Intended use.** Research on sentence simplification. Outputs are not checked for factual
accuracy; do not use them for patient-facing or other high-stakes text without expert review.

**License.** Use of this adapter is subject to the license of the base model (MIT)
and of the training dataset.
