# Nirṇaya — Inference

**Nirṇaya** ("निर्णय", *decision*) is an India-first **System-One decision model**. Given a `state` (text — often Indic-language, code-mixed, or Hinglish) plus one or more **typed questions**, Nirṇaya returns **calibrated probabilities in a single forward pass**, with no text generation.

This is the inference package. The model weights live on Hugging Face at [`cmul8-hf/nirnaya`](https://huggingface.co/cmul8-hf/nirnaya).

## Install

**Prerequisites:** Python ≥ 3.10 and a CUDA-capable GPU with bf16 support (Ampere or newer).

### 1. PyTorch

Pick the CUDA wheel that matches your driver 

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu118 / cu121 / cu126 / cu128
```

### 2. Nirṇaya

```bash
pip install git+https://github.com/cmul8/nirnaya.git@v0.1.0
```

### 3. Verify

```bash
python -c "import nirnaya; print(nirnaya.__version__)"   # -> 0.1.0
```

### First-run model download

The first `NirnayaModel.from_pretrained("cmul8-hf/nirnaya")` call downloads:

- **Qwen3-4B backbone** — ~8 GB, one-time
- **Nirṇaya adapter + heads** — ~200 MB

### For contributors

```bash
git clone https://github.com/cmul8/nirnaya.git
cd nirnaya
pip install -e ".[dev]"
pytest tests/ -v
```

## Quickstart

```python
from nirnaya import NirnayaModel

model = NirnayaModel.from_pretrained("cmul8-hf/nirnaya")
result = model.decide(
    state="बिजली बिल का पेमेंट अभी तक नहीं हुआ, urgent hai please",
    questions=[
        {"type": "choice", "options": ["billing", "technical", "sales"]},
        {"type": "score",  "k": 5, "instructions": "How urgent is this?"},
        {"type": "noul",   "instructions": "Should this be escalated?"},
    ],
)
print(result[0].probs)            # [0.87, 0.09, 0.04]
print(result[1].expected_level)   # 4.2
print(result[2].p_true)           # 0.71
```

All three questions above run in **a single forward pass**.

## Question shapes

| Type | Required fields | Optional | Result |
|---|---|---|---|
| `choice` | `options: list[str]` | `instructions: str` | `probs: list[float]` sums to 1; `answer: int` = argmax |
| `score` | `options: list[str]` **or** `k: int` (2–64) | `instructions: str` | `probs`, `answer`, plus `expected_level: float` (1-indexed mean) |
| `noul` | — | `instructions: str` | `probs = [P(false), P(true)]`, `answer: bool`, `p_true: float` |

Every `Decision` also carries a `confidence: float ∈ [0, 1]` calibrated so 1.0 is a fully certain top pick and 0.0 is uniform.

## Licence

Weights and code are released under **CC BY-NC 4.0** — free for research, evaluation, and personal use. 

The base model, [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), is Apache-2.0 and its terms continue to apply to the base weights.

## Citation

```bibtex
@misc{nirnaya2026v1,
  title  = {Nirṇaya: An India-first System-One Decision Model},
  author = {Ghosh, Basab and Sarkale, Sagar and Katte, Abhijeet},
  year   = {2026},
  url    = {https://huggingface.co/cmul8-hf/nirnaya},
  note   = {v1.0}
}
```
