"""GRPO reward: generated search queries -> retrieve -> recall/MRR of next basket.

Two matching modes:
  match="exact" — credit only when the retrieved set contains the customer's actual
                  next article_ids (sparse: ~105K items, most groups score 0).
  match="type"  — credit when retrieved items share the next basket's product *type*
                  (dense: many items per type -> far more groups carry signal).

Returns a function with TRL's GRPOTrainer signature:
    reward_func(completions, **kwargs) -> list[float]
The dataset's `basket` / `basket_types` columns arrive as kwargs.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    """Fixed catalog index — embeds queries, returns nearest article ids (and types)."""

    def __init__(self, vec_path, id_path, type_path=None, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.vecs = np.load(vec_path)
        self.ids = np.load(id_path, allow_pickle=True).tolist()
        self.id2type = {}
        if type_path:
            types = np.load(type_path, allow_pickle=True).tolist()
            self.id2type = dict(zip(self.ids, types))

    def retrieve(self, query: str, k: int) -> list[str]:
        qv = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vecs @ qv
        return [self.ids[i] for i in np.argsort(-sims)[:k]]


def parse_queries(text: str, cap: int = 5) -> list[str]:
    lines = [ln.strip(" -•\t0123456789.") for ln in text.splitlines() if ln.strip()]
    return (lines or [text.strip()])[:cap]


def make_reward_fn(retriever: Retriever, k=20, match="exact", query_budget=3,
                   cand_budget=60, lam_q=0.10, lam_c=0.005):
    """Build the GRPO reward. `match`: 'exact' (article ids) or 'type' (product types)."""
    def reward_func(completions, basket, basket_types=None, **kwargs) -> list[float]:
        basket_types = basket_types or [None] * len(completions)
        rewards = []
        for comp, ids, types in zip(completions, basket, basket_types):
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            queries = parse_queries(text, cap=query_budget + 2)

            best_rank: dict[str, int] = {}          # article_id -> best rank across queries
            for q in queries:
                for rank, iid in enumerate(retriever.retrieve(q, k), start=1):
                    if iid not in best_rank or rank < best_rank[iid]:
                        best_rank[iid] = rank

            if match == "type":
                by_type: dict[str, int] = {}        # product_type -> best rank of any item
                for iid, rank in best_rank.items():
                    t = retriever.id2type.get(iid)
                    if t is not None and (t not in by_type or rank < by_type[t]):
                        by_type[t] = rank
                targets, ranks = types, by_type
            else:
                targets, ranks = ids, best_rank

            mrr = float(np.mean([1.0 / ranks[t] if t in ranks else 0.0 for t in targets]))
            penalty = (lam_q * max(0, len(queries) - query_budget)
                       + lam_c * max(0, len(best_rank) - cand_budget))
            rewards.append(mrr - penalty)
        return rewards

    return reward_func
