"""One-time local prep — produces the small artifacts to upload to a GPU box.

Reads the big transactions_train.csv LOCALLY only; the GPU box (Colab) then needs
just these files (never the 3.5 GB CSV):
    data/dataset.jsonl            train prompts + baskets
    data/eval.jsonl               held-out prompts + baskets (disjoint customers)
    data/catalog_vecs_minilm.npy  retriever index (~160 MB)
    data/catalog_ids.npy
    data/catalog_types.npy        per-item product type (type-level reward)
"""
import os

from src.data import (ID_CACHE, TYPE_CACHE, VEC_CACHE, build_dataset,
                      embed_catalog)

HERE = os.path.dirname(__file__)
TRAIN_OUT = os.path.join(HERE, "data/dataset.jsonl")
EVAL_OUT = os.path.join(HERE, "data/eval.jsonl")


def main():
    embed_catalog()                                  # caches vecs + ids + types
    train = build_dataset(n_customers=2000, skip=0)
    eval_ = build_dataset(n_customers=300, skip=2000)  # disjoint held-out customers
    train.to_json(TRAIN_OUT)
    eval_.to_json(EVAL_OUT)
    print(f"\ntrain {len(train)} -> {TRAIN_OUT}")
    print(f"eval  {len(eval_)} -> {EVAL_OUT}")
    print("upload to Colab:", TRAIN_OUT, EVAL_OUT, VEC_CACHE, ID_CACHE, TYPE_CACHE)


if __name__ == "__main__":
    main()
