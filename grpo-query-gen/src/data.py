"""Build the GRPO prompt dataset from H&M, and (one-time) embed the catalog.

Each training example:
    prompt  = chat messages: system + customer's recent-purchase history
    basket  = list of article_ids the customer actually bought next (reward target)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.dirname(__file__))
ART = os.path.join(HERE, "data/raw/articles.csv")
TXN = os.path.join(HERE, "data/raw/transactions_train.csv")
VEC_CACHE = os.path.join(HERE, "data/catalog_vecs_minilm.npy")
ID_CACHE = os.path.join(HERE, "data/catalog_ids.npy")

SYSTEM = (
    "You are a fashion search assistant. Given a customer's recent purchases, "
    "output up to 3 short search queries (one per line) for items they're likely "
    "to buy next. Output only the queries."
)


def _catalog_frame() -> pd.DataFrame:
    cols = ["article_id", "prod_name", "product_type_name", "product_group_name",
            "colour_group_name", "detail_desc"]
    df = pd.read_csv(ART, usecols=cols, dtype={"article_id": str}).fillna("")
    df["text"] = (df.prod_name + " — " + df.product_type_name + ", " + df.colour_group_name
                  + ", " + df.product_group_name + ". " + df.detail_desc).str.slice(0, 300)
    return df


def embed_catalog(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    """One-time: embed the whole catalog and cache vectors + ids."""
    if os.path.exists(VEC_CACHE):
        return
    df = _catalog_frame()
    model = SentenceTransformer(model_name)
    vecs = model.encode(df.text.tolist(), batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    np.save(VEC_CACHE, vecs)
    np.save(ID_CACHE, np.array(df.article_id.tolist(), dtype=object))


def build_dataset(n_customers=2000, hist_n=6, basket_n=4, nrows=5_000_000) -> Dataset:
    cat = _catalog_frame().set_index("article_id")
    name = cat.prod_name.to_dict()

    txn = pd.read_csv(TXN, usecols=["t_dat", "customer_id", "article_id"],
                      dtype={"article_id": str, "customer_id": str}, nrows=nrows)
    txn = txn.sort_values("t_dat")

    rows = []
    for _, g in txn.groupby("customer_id", sort=False):
        arts = list(dict.fromkeys(g.article_id.tolist()))
        if len(arts) < hist_n + basket_n:
            continue
        history, basket = arts[-(hist_n + basket_n):-basket_n], arts[-basket_n:]
        hist_str = ", ".join(name.get(a, a) for a in history)
        rows.append({
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": f"Recent purchases: {hist_str}"}],
            "basket": [b for b in basket if b in cat.index],
        })
        if len(rows) >= n_customers:
            break
    return Dataset.from_list([r for r in rows if r["basket"]])
