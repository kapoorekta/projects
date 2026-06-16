"""GRPO reward: generated search queries -> retrieve -> recall/MRR of next basket.

Exposes make_reward_fn() returning a function with TRL's GRPOTrainer signature:
    reward_func(completions, **kwargs) -> list[float]
where the dataset's `basket` column arrives as a kwarg aligned with the batch.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    """Fixed catalog index — embeds queries and returns nearest article ids."""

    def __init__(self, vec_path: str, id_path: str, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.vecs = np.load(vec_path)
        self.ids = np.load(id_path, allow_pickle=True).tolist()

    def retrieve(self, query: str, k: int) -> list[str]:
        qv = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vecs @ qv
        return [self.ids[i] for i in np.argsort(-sims)[:k]]


def parse_queries(text: str, cap: int = 5) -> list[str]:
    """Split a completion into individual queries (one per line)."""
    lines = [ln.strip(" -•\t0123456789.") for ln in text.splitlines() if ln.strip()]
    return (lines or [text.strip()])[:cap]


def make_reward_fn(retriever: Retriever, k=20, query_budget=3, cand_budget=60,
                   lam_q=0.10, lam_c=0.005):
    # cand_budget ~= query_budget * k so a normal 3-query answer isn't penalized;
    # lam_q is what actually discourages brute-forcing many queries.
    """Build the GRPO reward function over a fixed retriever."""
    def reward_func(completions, basket, **kwargs) -> list[float]:
        rewards = []
        for comp, targets in zip(completions, basket):
            text = comp if isinstance(comp, str) else comp[-1]["content"]  # chat or plain
            queries = parse_queries(text, cap=query_budget + 2)
            best_rank: dict[str, int] = {}
            for q in queries:
                for rank, iid in enumerate(retriever.retrieve(q, k), start=1):
                    if iid not in best_rank or rank < best_rank[iid]:
                        best_rank[iid] = rank
            mrr = float(np.mean([1.0 / best_rank[t] if t in best_rank else 0.0 for t in targets]))
            penalty = (lam_q * max(0, len(queries) - query_budget)
                       + lam_c * max(0, len(best_rank) - cand_budget))
            rewards.append(mrr - penalty)
        return rewards

    return reward_func
