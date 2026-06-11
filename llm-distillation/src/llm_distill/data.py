"""Load the Amazon ESCI dataset and turn it into clean (query, product, label)
examples ready for teacher labeling / student training.

Stage output: data/interim/{train,eval}.jsonl  — one example per line:
    {"id", "query", "product": {...}, "gold_label": "E|S|C|I"}
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .prompts import RAW_TO_CODE
from .utils import get_logger, write_jsonl

log = get_logger(__name__)

PRODUCT_FIELDS = [
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


def _to_example(row: dict, idx: int, max_chars: int) -> dict | None:
    raw_label = row.get("esci_label")
    code = RAW_TO_CODE.get(str(raw_label))
    if code is None:
        return None
    query = (row.get("query") or "").strip()
    if not query:
        return None
    product = {f: (row.get(f) or "")[:max_chars] for f in PRODUCT_FIELDS}
    return {
        "id": str(row.get("example_id", row.get("product_id", idx))),
        "query": query,
        "product": product,
        "gold_label": code,
    }


def _keep(row: dict, d) -> bool:
    """Filter to the chosen locale and (optionally) the smaller task-1 subset."""
    if d.locale and str(row.get("product_locale", "")).lower() != d.locale.lower():
        return False
    if d.use_small_version and "small_version" in row and not row.get("small_version"):
        return False
    return True


def prepare(cfg: Config) -> dict[str, Path]:
    """Download/stream, filter, sample, and write interim train/eval splits."""
    from datasets import load_dataset

    d = cfg.data
    n_total = d.n_train + d.n_eval
    log.info("Loading %s (locale=%s, small=%s, streaming=%s)",
             d.dataset_id, d.locale, d.use_small_version, d.streaming)

    ds = load_dataset(d.dataset_id, split="train", streaming=d.streaming)
    ds = ds.filter(lambda row: _keep(row, d))
    ds = ds.shuffle(seed=cfg.seed, buffer_size=d.shuffle_buffer) if d.streaming \
        else ds.shuffle(seed=cfg.seed)

    # Take only what we need. Streaming yields lazily, so we never pull the full set.
    rows = ds.take(n_total) if d.streaming else ds.select(range(min(n_total, len(ds))))

    examples = []
    for i, row in enumerate(rows):
        ex = _to_example(row, i, d.max_product_chars)
        if ex is not None:
            examples.append(ex)

    train_rows = examples[: d.n_train]
    eval_rows = examples[d.n_train : d.n_train + d.n_eval]

    interim = Path(cfg.paths.interim_dir)
    out = {
        "train": interim / "train.jsonl",
        "eval": interim / "eval.jsonl",
    }
    n_tr = write_jsonl(out["train"], train_rows)
    n_ev = write_jsonl(out["eval"], eval_rows)
    log.info("Wrote %d train / %d eval examples to %s", n_tr, n_ev, interim)
    _log_label_balance(train_rows)
    return out


def _log_label_balance(rows: list[dict]) -> None:
    from collections import Counter

    counts = Counter(r["gold_label"] for r in rows)
    log.info("Train label balance: %s", dict(sorted(counts.items())))
