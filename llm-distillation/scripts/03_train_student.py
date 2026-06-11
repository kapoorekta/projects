#!/usr/bin/env python
"""Fine-tune the student with QLoRA on the teacher-labeled data."""
import argparse

from dotenv import load_dotenv

from llm_distill.config import load_config
from llm_distill.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
