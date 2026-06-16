# GRPO for LLM Query Generation (H&M Retrieval)

Train a small LLM with **GRPO** (the RL algorithm behind DeepSeek-R1) to generate
**search queries** from a customer's purchase history — rewarded by whether those
queries actually *retrieve* the items the customer buys next.

This is the **LLM candidate-generation → retrieval** pattern used in modern search
and recommendation: an LLM expands intent into queries, a retriever turns them into
candidates. Reframing recommendation as *query generation* makes it a true text-
generation task with a **verifiable, retrieval-grounded reward** — exactly GRPO's
sweet spot. Data: H&M's [Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

## Why query generation (not item-ID prediction)

GRPO optimizes a *generative* policy against a reward. Predicting item IDs is a
ranking/retrieval problem (awkward for GRPO); **generating queries** is natural
language generation with a reward you can verify against ground truth:

> reward = does the retriever, fed the model's queries, surface the customer's
> *actual* next purchases?

Grounding the reward in real retrieval (not free-text similarity) makes it
**verifiable and hard to hack** — a vague query simply won't retrieve the right item.

## How it works

```
customer history ──▶ LLM generates search queries
                          │
                   retrieve top-k from the 105K-item catalog (MiniLM + cosine)
                          │
        reward = MRR / recall of the next-7-day basket  −  anti-brute-force penalty
                          │
                   GRPO update (group-relative advantage, KL to reference)
```

- **Group-relative advantage:** sample G queries per prompt, score each, advantage = reward − group mean (no critic).
- **Reward** (`src/reward.py`): retrieve per query → union candidates → recall/MRR of the real next basket, minus a penalty that stops the model brute-forcing many broad queries.
- **Retriever:** the 105K H&M catalog embedded once with a free local `all-MiniLM-L6-v2` (the reward runs every rollout, so a local embedder — not an API — is essential).

## Project structure

```
grpo-query-gen/
├── src/
│   ├── reward.py          # Retriever + GRPO reward (queries → retrieve → recall/MRR)
│   └── data.py            # catalog embedding (cached) + prompt-dataset builder
├── train_grpo.py          # GRPO fine-tuning via trl.GRPOTrainer
├── reward_prototype.py    # reward-design sandbox (tiny mock catalog)
├── real_reward_test.py    # reward validation on the real H&M catalog
└── data/                  # H&M CSVs + cached embeddings (git-ignored)
```

## Reward design — validated before training

The reward was prototyped and checked on real H&M data *before* wiring the RL loop
(the part most likely to make or break the project). It cleanly separates good
queries from the failure modes:

| Query style | Outcome |
|---|---|
| targeted, on-intent | high reward (retrieves the next items) |
| generic ("clothing") | ~0 |
| echoing the purchase history | ~0 (past ≠ next) |
| brute-force spam | high raw recall, but penalty drops it below honest answers |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers pandas datasets        # reward + data
pip install trl peft transformers                         # GRPO (GPU)

python real_reward_test.py     # validate the reward on real customers (CPU, free)
python train_grpo.py           # GRPO fine-tune — needs a GPU (Colab T4)
```

GRPO is generation-heavy (G completions per prompt), so training runs on a GPU.
The dataset is tiny, so for Colab you ship the prebuilt dataset + cached catalog
embeddings — not the 3.5 GB transactions file.

## Tech stack

Python · Hugging Face `trl` (GRPO) / `transformers` / `peft` ·
`sentence-transformers` (MiniLM retriever) · H&M Kaggle dataset.

> **Status:** reward designed + validated on real data, GRPO loop wired. Training
> (Colab) and base-vs-GRPO recall@k evaluation are the next step.
