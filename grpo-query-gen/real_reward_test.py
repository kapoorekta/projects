"""Run the query-generation reward on the REAL H&M catalog + customers,
using a FREE local embedder (MiniLM) — the right choice for GRPO, where the
reward embeds queries on every rollout (an API would be slow/rate-limited/costly).

  catalog   = articles.csv (enriched text), embedded once and cached
  sequences = transactions_train.csv (sampled customers)
  reward    = MRR/recall of the next basket in the retrieved candidate set
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "data/raw/articles.csv")
TXN = os.path.join(HERE, "data/raw/transactions_train.csv")
VEC_CACHE = os.path.join(HERE, "data/catalog_vecs_minilm.npy")
ID_CACHE = os.path.join(HERE, "data/catalog_ids.npy")

_model = SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    return _model.encode(texts, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=len(texts) > 5000).astype(np.float32)


def build_catalog() -> tuple[list[str], list[str]]:
    cols = ["article_id", "prod_name", "product_type_name", "product_group_name",
            "colour_group_name", "detail_desc"]
    df = pd.read_csv(ART, usecols=cols, dtype={"article_id": str}).fillna("")
    text = (df.prod_name + " — " + df.product_type_name + ", " + df.colour_group_name
            + ", " + df.product_group_name + ". " + df.detail_desc).str.slice(0, 300)
    return df.article_id.tolist(), text.tolist()


print("Building catalog from articles.csv ...", flush=True)
CAT_IDS, CAT_TEXTS = build_catalog()
ID2TEXT = dict(zip(CAT_IDS, CAT_TEXTS))

if os.path.exists(VEC_CACHE):
    print("Loading cached MiniLM catalog embeddings ...", flush=True)
    CAT_VECS = np.load(VEC_CACHE)
    CAT_IDS = np.load(ID_CACHE, allow_pickle=True).tolist()
else:
    print(f"Embedding {len(CAT_TEXTS)} items with MiniLM (one-time) ...", flush=True)
    CAT_VECS = embed(CAT_TEXTS)
    np.save(VEC_CACHE, CAT_VECS)
    np.save(ID_CACHE, np.array(CAT_IDS, dtype=object))
print(f"Catalog ready: {len(CAT_IDS)} items, dim {CAT_VECS.shape[1]}\n", flush=True)


def retrieve(query: str, k: int) -> list[str]:
    qv = embed([query])[0]
    sims = CAT_VECS @ qv
    return [CAT_IDS[i] for i in np.argsort(-sims)[:k]]


def reward(queries, target_ids, k=20, query_budget=3, cand_budget=40,
           lam_q=0.10, lam_c=0.005) -> dict:
    best_rank: dict[str, int] = {}
    for q in queries:
        for rank, iid in enumerate(retrieve(q, k), start=1):
            if iid not in best_rank or rank < best_rank[iid]:
                best_rank[iid] = rank
    mrr = float(np.mean([1.0 / best_rank[t] if t in best_rank else 0.0 for t in target_ids]))
    recall = float(np.mean([1.0 if t in best_rank else 0.0 for t in target_ids]))
    n_cand = len(best_rank)
    penalty = lam_q * max(0, len(queries) - query_budget) + lam_c * max(0, n_cand - cand_budget)
    return {"reward": round(mrr - penalty, 3), "mrr": round(mrr, 3),
            "recall": round(recall, 2), "n_cand": n_cand}


def build_sequences(n=4, hist_n=6, basket_n=3, nrows=3_000_000):
    df = pd.read_csv(TXN, usecols=["t_dat", "customer_id", "article_id"],
                     dtype={"article_id": str, "customer_id": str}, nrows=nrows)
    df = df.sort_values("t_dat")
    seqs = []
    for _, g in df.groupby("customer_id", sort=False):
        arts = list(dict.fromkeys(g.article_id.tolist()))
        if len(arts) < hist_n + basket_n:
            continue
        seqs.append((arts[-(hist_n + basket_n):-basket_n], arts[-basket_n:]))
        if len(seqs) >= n:
            break
    return seqs


if __name__ == "__main__":
    print("Building a few real customer sequences ...\n", flush=True)
    for i, (history, basket) in enumerate(build_sequences(), 1):
        print(f"===== customer {i} =====")
        print("history:", [ID2TEXT[a].split(" — ")[0] for a in history])
        print("next basket:", [ID2TEXT[a].split(" — ")[0] for a in basket])
        oracle = " ".join(ID2TEXT[b].split(".")[0] for b in basket)
        scenarios = {
            "oracle (basket types)": [oracle],
            "history-derived guess": [" ".join(ID2TEXT[h].split(" — ")[0] for h in history)],
            "generic": ["clothing fashion items"],
        }
        for name, qs in scenarios.items():
            r = reward(qs, basket)
            print(f"  {name:<24} reward={r['reward']:>6}  mrr={r['mrr']:>5}  "
                  f"recall={r['recall']:>4}  cands={r['n_cand']}")
        print()
