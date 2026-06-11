#!/usr/bin/env python
"""Score student (and optionally teacher) vs gold ESCI labels -> metrics.json."""
import argparse

from dotenv import load_dotenv

from llm_distill.config import load_config
from llm_distill.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
