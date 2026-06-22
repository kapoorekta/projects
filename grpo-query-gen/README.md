# GRPO for LLM Query Generation (H&M Retrieval)

This project trains a small language model with **Group Relative Policy
Optimization (GRPO)** — the reinforcement-learning algorithm introduced with
DeepSeek-R1 — to generate **search queries** from a customer's purchase history.
Queries are rewarded according to whether the items they retrieve include the
customer's subsequent purchases.

The formulation follows the **LLM candidate-generation → retrieval** pattern used
in modern search and recommendation systems: a language model expands user intent
into queries, and a retriever converts those queries into candidate items. Framing
recommendation as query generation produces a text-generation task with a
**verifiable, retrieval-grounded reward**, which is well suited to GRPO. Dataset:
H&M [Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

## Problem formulation

GRPO optimizes a generative policy against a scalar reward. Predicting item
identifiers directly is a ranking problem and is poorly matched to a generative
objective; generating queries is a natural-language generation task whose output
can be scored against ground truth:

> reward = whether the retriever, given the model's queries, surfaces the
> customer's actual next purchases.

Grounding the reward in retrieval outcomes rather than free-text similarity keeps
it verifiable and resistant to reward hacking, as a non-specific query will not
retrieve the correct item.

## Method

```
customer history ──▶ LLM generates search queries
                          │
                   retrieve top-k from the 105K-item catalog (MiniLM + cosine)
                          │
        reward = MRR / recall of the next-7-day basket  −  anti-brute-force penalty
                          │
                   GRPO update (group-relative advantage, KL to reference)
```

- **Group-relative advantage:** for each prompt, G queries are sampled and scored;
  each sample's advantage is its reward minus the group mean, removing the need for
  a value/critic network.
- **Reward** (`src/reward.py`): each query is retrieved against the catalog; the
  union of candidates is scored by reciprocal rank / recall of the actual next
  basket, less a penalty that discourages brute-force query enumeration.
- **Retriever:** the 105K-item H&M catalog is embedded once with a local
  `all-MiniLM-L6-v2` model. A local embedder is required because the reward is
  computed on every rollout, which precludes an external API.

## Repository structure

```
grpo-query-gen/
├── src/
│   ├── reward.py          # Retriever + GRPO reward (queries → retrieve → recall/MRR)
│   └── data.py            # catalog embedding (cached) + prompt-dataset builder
├── train_grpo.py          # GRPO fine-tuning via trl.GRPOTrainer
├── reward_prototype.py    # reward-design sandbox (mock catalog)
├── real_reward_test.py    # reward validation on the real H&M catalog
└── data/                  # H&M CSVs + cached embeddings (git-ignored)
```

## Reward validation

The reward function was prototyped and verified on real H&M data prior to training,
since reward design is the principal determinant of success. The reward cleanly
distinguishes effective queries from common failure modes:

| Query style | Outcome |
|---|---|
| Targeted, on-intent | High reward (retrieves the next items) |
| Generic | Near zero |
| Repetition of purchase history | Near zero (past purchases differ from next) |
| Brute-force enumeration | High raw recall, reduced below honest answers by the penalty |

## Experiments and results

Models were evaluated by recall@20 of each customer's next-7-day basket on
held-out customers, computed from the model's generated queries. The project
progressed through four reward formulations:

| Reward design | Model | recall@20 (base → GRPO) | Observation |
|---|---|---|---|
| Exact-article match | 0.5B | not evaluated | Reward ≈ 0 with ~90% of prompts at zero reward variance → no usable gradient |
| Product-type match | 0.5B | 0.036 → 0.031 | Dense signal, but a proxy: optimizing type did not improve exact recall |
| Graded (exact within top-k + type) | 0.5B | — | The exact term stayed ≈ 0 (rarely within top-k), so it contributed no signal |
| **Graded + rank-graded exact** | **1.5B** | **0.022 → 0.032 (+46%)** | Aligned, varying signal produced a measurable improvement |

The final configuration improved recall@20 by approximately **46% relative** to the
base model. Note that the 1.5B base underperformed the 0.5B base (it produced
longer, less precise queries); part of GRPO's improvement corresponds to correcting
this. Additional model capacity was beneficial only after reinforcement learning.

## Findings

Reward design, rather than model size or hyperparameters, governed the outcome of
every run:

1. **Sparse rewards provide no learning signal.** Rewarding exact-article retrieval
   (one item among 105K) yielded approximately zero reward on nearly all samples,
   leaving no gradient.
2. **A dense proxy optimizes the wrong objective.** A product-type reward trained
   successfully but did not improve exact recall: the policy optimizes precisely
   the quantity it is scored on.
3. **A reward term contributes only if it varies across samples.** A 50/50
   combination of exact and type rewards still failed, because the exact term was
   uniformly zero; GRPO derives its signal from differences among samples, so a
   constant term has no effect regardless of its weight.
4. **Grading the aligned term by rank restores signal.** Scoring the target item by
   its rank within a deep (top-200) retrieval made the exact term non-zero and
   variable, after which recall improved.
5. **Larger models are not unconditionally better.** The 1.5B base underperformed
   the 0.5B base zero-shot; the additional capacity was advantageous only after
   reinforcement learning constrained the output.

In summary, GRPO optimizes exactly the supplied reward. An effective reward must be
both **dense** (varying across samples) and **aligned** (consistent with the
evaluation metric); satisfying both simultaneously is the central difficulty.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers pandas datasets   # reward + data
pip install trl peft transformers                    # GRPO (GPU)

python real_reward_test.py     # validate the reward on real customers (CPU)
python train_grpo.py           # GRPO fine-tuning (GPU; e.g. Colab T4)
python eval.py                 # recall@20, base vs GRPO
```

GRPO is generation-heavy (G completions per prompt) and requires a GPU for
training. The dataset is small, so a GPU host needs only the prebuilt dataset and
cached catalog embeddings, not the 3.5 GB transactions file.

## Stack

Python · Hugging Face `trl` (GRPO), `transformers`, `peft` ·
`sentence-transformers` (MiniLM retriever) · H&M Kaggle dataset.
