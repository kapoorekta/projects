"""One-time local prep — produces the small artifacts to upload to a GPU box.

Reads the big transactions_train.csv LOCALLY only; the GPU box (Colab) then needs
just these three files (never the 3.5 GB CSV):
    data/dataset.jsonl            prompts + baskets (KB)
    data/catalog_vecs_minilm.npy  retriever index (~160 MB)
    data/catalog_ids.npy
"""
import os

from src.data import ID_CACHE, VEC_CACHE, build_dataset, embed_catalog

OUT = os.path.join(os.path.dirname(__file__), "data/dataset.jsonl")


def main():
    embed_catalog()                          # caches catalog vectors (skips if present)
    ds = build_dataset(n_customers=2000)
    ds.to_json(OUT)
    print(f"\nwrote {len(ds)} examples -> {OUT}")
    print("upload to Colab:", OUT, VEC_CACHE, ID_CACHE)


if __name__ == "__main__":
    main()
