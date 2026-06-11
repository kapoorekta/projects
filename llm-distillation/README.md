# LLM Distillation

Distill a large **teacher** LLM into a small **student** for Amazon **ESCI**
search-relevance classification (label each query–product pair `E`xact /
`S`ubstitute / `C`omplement / `I`rrelevant).

The student is far smaller and cheaper to run than the teacher, but learns from
the teacher's outputs to approach its quality — on a task with **built-in ground
truth** (the gold ESCI labels), so you can objectively measure whether it worked.

## Layout

```
llm-distillation/
├── configs/config.yaml        # every knob for every stage
├── src/llm_distill/
│   ├── config.py              # YAML -> typed dataclasses
│   ├── prompts.py             # label space + prompt templates (shared!)
│   ├── data.py                # stage 1: load + sample ESCI
│   ├── teacher.py             # stage 2: teacher labels + rationales
│   ├── train.py               # stage 3: QLoRA SFT of the student
│   ├── evaluate.py            # stage 4: score vs gold labels
│   └── utils.py               # logging, seeding, JSONL I/O
├── scripts/0{1..4}_*.py       # thin CLI wrappers, each takes --config
├── tests/                     # fast, offline unit tests
├── data/                      # raw/interim/processed (git-ignored)
└── artifacts/                 # trained adapters + metrics (git-ignored)
```

## The pipeline

```
ESCI (HF)  ──①prepare──▶  interim/{train,eval}.jsonl
                              │
                  ②teacher labels train split
                              ▼
                processed/train_labeled.jsonl
                              │
                     ③QLoRA fine-tune student
                              ▼
                  artifacts/student-qlora/  ──④evaluate vs gold──▶ metrics.json
```

1. **Prepare** — download ESCI, filter locale, sample N examples, write JSONL.
2. **Teacher** — a big model labels each pair as E/S/C/I *with a rationale*.
   Keep the gold label and distill the *reasoning* (default), or let the teacher
   predict the label itself (`teacher.keep_gold_label: false`).
3. **Train** — QLoRA-fine-tune a small instruct model on the teacher outputs.
4. **Evaluate** — run the student on the held-out split and compare its
   predictions to the **gold ESCI labels** (accuracy + macro-F1). Optionally
   score the teacher too, to see how much quality the student retained.

## Quickstart

```bash
cd llm-distillation
python -m venv .venv && source .venv/bin/activate

# Laptop / no GPU — enough for stages 1, 2, 4 and the tests:
pip install -e ".[teacher,dev]"
# Full run including QLoRA training (needs a CUDA GPU):
pip install -e ".[teacher,train,dev]"

cp .env.example .env        # add your OPENAI_API_KEY (or ANTHROPIC_API_KEY)

make test                   # offline sanity check, no keys needed
make data                   # 1. download + sample ESCI
make teacher                # 2. generate teacher labels   (uses API credits)
make train                  # 3. fine-tune student         (needs GPU)
make eval                   # 4. score vs gold labels
# or: make all
```

Everything is driven by `configs/config.yaml`. To run an experiment, copy it
(`cp configs/config.yaml configs/exp1.yaml`), edit, and pass
`--config configs/exp1.yaml` (or `make CONFIG=configs/exp1.yaml data`).
