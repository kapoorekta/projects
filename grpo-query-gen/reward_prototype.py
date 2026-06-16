"""Prototype the GRPO reward for query-generation recommendation.

Idea: the model reads a customer's purchase history and generates search
queries; a retriever turns those queries into candidate items; the reward is
how well those candidates cover the customer's *actual* next-basket purchases.

    queries --embed--> retrieve top-k from catalog --> union candidate set
    reward = MRR/recall of the next-basket items in that set  -  anti-brute-force penalty

This file uses a tiny MOCK catalog + OpenAI embeddings so we can eyeball what
scores well BEFORE training. For real GRPO:
  - swap CATALOG/HISTORY/NEXT_BASKET for H&M articles.csv / transactions.csv
  - swap the embedder for a local sentence-transformers model (no API per rollout)
The reward logic stays the same.
"""
from __future__ import annotations

import numpy as np
from openai import OpenAI

client = OpenAI()
EMB_MODEL = "text-embedding-3-small"

# --- mock catalog (id, descriptive text) — stand-in for H&M articles --------
CATALOG = [
    (1, "White Mesh Sneakers — casual everyday trainers, white"),
    (2, "Canvas Low-Top Sneakers — minimalist casual shoes, off-white"),
    (3, "Running Shoes — lightweight athletic trainers"),
    (4, "Black Leather Ankle Boots — formal boots"),
    (5, "Chelsea Boots — suede slip-on boots"),
    (6, "Strappy Heels — evening shoes"),
    (7, "Leather Sandals — summer flat sandals"),
    (8, "Slim Fit Jeans — blue denim trousers"),
    (9, "Skinny Jeans — black denim"),
    (10, "Chino Trousers — beige cotton pants"),
    (11, "Pullover Hoodie — grey cotton hooded sweatshirt"),
    (12, "Crewneck Sweatshirt — navy fleece"),
    (13, "White Cotton T-Shirt — basic crew tee"),
    (14, "Graphic Print T-Shirt — black cotton tee"),
    (15, "Oxford Shirt — formal button-down"),
    (16, "Puffer Jacket — quilted winter coat"),
    (17, "Denim Jacket — casual blue jean jacket"),
    (18, "Wool Overcoat — formal long coat"),
    (19, "Floral Summer Dress — light midi dress"),
    (20, "Knit Sweater Dress — winter dress"),
    (21, "Pleated Skirt — midi skirt"),
    (22, "Cotton Crew Socks — pack of trainer socks, white"),
    (23, "Baseball Cap — casual cotton cap"),
    (24, "Beanie Hat — knit winter hat"),
    (25, "Wool Gloves — winter gloves"),
    (26, "Canvas Backpack — casual daypack"),
    (27, "Leather Belt — brown formal belt"),
    (28, "Sunglasses — round metal frames"),
]

# --- one mock customer (history + the basket they ACTUALLY bought next) -----
HISTORY = ["Slim Fit Jeans", "Pullover Hoodie", "White Cotton T-Shirt", "Puffer Jacket"]
NEXT_BASKET_IDS = [1, 2, 22, 23]  # white sneakers, canvas sneakers, socks, cap

NAME = {cid: text for cid, text in CATALOG}


def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    v = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # normalize for cosine


CAT_IDS = [c[0] for c in CATALOG]
CAT_VECS = embed([c[1] for c in CATALOG])


def retrieve(query: str, k: int) -> list[int]:
    qv = embed([query])[0]
    sims = CAT_VECS @ qv
    return [CAT_IDS[i] for i in np.argsort(-sims)[:k]]


def reward(queries, target_ids, k=5, query_budget=3, cand_budget=12,
           lam_q=0.10, lam_c=0.02) -> dict:
    """queries -> retrieve -> reward = MRR over the next-basket - brute-force penalty."""
    best_rank: dict[int, int] = {}
    for q in queries:
        for rank, iid in enumerate(retrieve(q, k), start=1):
            if iid not in best_rank or rank < best_rank[iid]:
                best_rank[iid] = rank
    mrr = float(np.mean([1.0 / best_rank[t] if t in best_rank else 0.0 for t in target_ids]))
    recall = float(np.mean([1.0 if t in best_rank else 0.0 for t in target_ids]))
    n_cand = len(best_rank)
    penalty = lam_q * max(0, len(queries) - query_budget) + lam_c * max(0, n_cand - cand_budget)
    return {"reward": round(mrr - penalty, 3), "mrr": round(mrr, 3),
            "recall": round(recall, 2), "penalty": round(penalty, 3),
            "n_queries": len(queries), "n_cand": n_cand}


SCENARIOS = {
    "targeted (good)":   ["white casual minimalist sneakers", "everyday comfortable canvas shoes"],
    "vague / generic":   ["clothing", "fashion items"],
    "off-topic":         ["wool winter gloves", "formal leather boots"],
    "echo the history":  ["slim jeans pullover hoodie white tee puffer jacket"],
    "brute-force spam":  ["sneakers", "boots", "socks", "cap", "shoes", "trainers", "hat", "shirt"],
}

if __name__ == "__main__":
    print("History:", HISTORY)
    print("Actual next basket:", [NAME[i] for i in NEXT_BASKET_IDS], "\n")
    hdr = f"{'scenario':<20}{'reward':>8}{'mrr':>7}{'recall':>8}{'penalty':>9}{'queries':>9}{'cands':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, qs in SCENARIOS.items():
        r = reward(qs, NEXT_BASKET_IDS)
        print(f"{name:<20}{r['reward']:>8}{r['mrr']:>7}{r['recall']:>8}"
              f"{r['penalty']:>9}{r['n_queries']:>9}{r['n_cand']:>7}")
