# LLM Distillation on Amazon ESCI

Distill a large teacher LLM into a small, deployable student for e-commerce
**search-relevance classification** — and measure exactly how much quality transfers.

A 0.5B open-source model is fine-tuned on a larger model's outputs to label
`(query, product)` pairs by relevance — **E**xact / **S**ubstitute /
**C**omplement / **I**rrelevant — on Amazon's [ESCI](https://github.com/amazon-science/esci-data)
benchmark. Because ESCI ships **gold labels**, the distilled student is scored
objectively against ground truth (accuracy + macro-F1), not vibes.

## Why this project

A demonstration of the full applied-distillation loop: turning an expensive
general model into a small, cheap, task-specific one. It exercises teacher data
generation, parameter-efficient fine-tuning (QLoRA), and evaluation design.

## Approach

Distillation comes in three forms: **logit/soft-label** (match the teacher's
probability distribution), **response** (learn from its answers), and
**rationale** (learn its reasoning). With a *closed-API* teacher there are no
logits and the tokenizers differ, so soft-label KD is off the table — this uses
**response + rationale distillation**:

1. **Teacher** (`gpt-4o-mini`) labels each pair `E/S/C/I` with a one-line rationale.
2. **Student** (`Qwen2.5-0.5B-Instruct`) is **QLoRA**-fine-tuned to reproduce
   `{label, rationale}`. Targets use the **gold label + the teacher's rationale**,
   so the student inherits *correct* answers plus the teacher's reasoning.
3. **Evaluate** the student against gold ESCI labels — accuracy, macro-F1, and a
   per-class confusion matrix.

## Pipeline

```
ESCI (HF) ──① prepare ──▶ interim/{train,eval}.jsonl
                              │
                  ② teacher labels (E/S/C/I + rationale)
                              ▼
                processed/train_labeled.jsonl
                              │
                  ③ QLoRA fine-tune Qwen2.5-0.5B
                              ▼
              artifacts/student-qlora ──④ evaluate vs gold──▶ metrics.json
```

Each stage is a script driven by `configs/config.yaml`, runnable independently.

## Project structure

```
llm-distillation/
├── configs/config.yaml        # single source of truth for every stage
├── src/llm_distill/
│   ├── config.py              # YAML → typed dataclasses
│   ├── prompts.py             # label space + shared prompt templates
│   ├── data.py                # ① stream + sample ESCI
│   ├── teacher.py             # ② teacher labelling (OpenAI / Anthropic)
│   ├── train.py               # ③ QLoRA SFT + class balancing
│   ├── evaluate.py            # ④ score vs gold labels
│   └── utils.py               # logging, seeding, JSONL I/O
├── scripts/0{1..4}_*.py       # CLI wrapper per stage (+ run_pipeline.py)
└── tests/                     # fast offline unit tests
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[teacher,train,dev]"   # drop 'train' if you have no GPU
cp .env.example .env                     # add OPENAI_API_KEY (or ANTHROPIC_API_KEY)

make test       # offline sanity check, no keys needed
make data       # ① stream + sample ESCI
make teacher    # ② generate teacher labels   (uses API credits)
make train      # ③ QLoRA fine-tune           (needs a CUDA GPU)
make eval       # ④ score vs gold labels
```

Everything is configured via `configs/config.yaml` — copy it and pass
`--config configs/your-exp.yaml` to run experiments without touching code.

## Tech stack

Python · Hugging Face `transformers` / `datasets` / `peft` / `trl` ·
`bitsandbytes` (4-bit QLoRA) · OpenAI API (teacher) · scikit-learn (metrics).
</content>
