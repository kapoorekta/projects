#!/usr/bin/env python
"""Label the training split with the teacher LLM -> data/processed/train_labeled.jsonl."""
import argparse

from dotenv import load_dotenv

from llm_distill.config import load_config
from llm_distill.teacher import label_training_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    label_training_set(cfg)


if __name__ == "__main__":
    main()
