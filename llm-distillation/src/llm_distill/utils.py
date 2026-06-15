"""Small shared helpers: logging, seeding, JSONL I/O."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence chatty third-party loggers (HF download/HTTP spam) — keep ours at INFO.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "datasets",
                  "urllib3", "filelock", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional: only if torch is installed
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
