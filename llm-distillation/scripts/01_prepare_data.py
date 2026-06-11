#!/usr/bin/env python
"""Download + sample ESCI into data/interim/{train,eval}.jsonl."""
import argparse

from dotenv import load_dotenv

from llm_distill.config import load_config
from llm_distill.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    prepare(cfg)


if __name__ == "__main__":
    main()
