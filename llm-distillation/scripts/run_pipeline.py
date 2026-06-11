#!/usr/bin/env python
"""Run the whole distillation pipeline (or a subset) in one command.

Handy on Colab/Kaggle where you want a single cell:

    python scripts/run_pipeline.py --config configs/config.yaml

Select / skip stages:

    python scripts/run_pipeline.py --stages data,teacher,eval   # baseline, no training
    python scripts/run_pipeline.py --skip data,teacher          # reuse cached data
"""
import argparse
import time

from dotenv import load_dotenv

from llm_distill.config import load_config

ALL_STAGES = ["data", "teacher", "train", "eval"]


def _run_stage(name: str, cfg) -> None:
    if name == "data":
        from llm_distill.data import prepare
        prepare(cfg)
    elif name == "teacher":
        from llm_distill.teacher import label_training_set
        label_training_set(cfg)
    elif name == "train":
        from llm_distill.train import train
        train(cfg)
    elif name == "eval":
        from llm_distill.evaluate import evaluate
        evaluate(cfg)
    else:
        raise ValueError(f"Unknown stage: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stages", help="comma-separated subset to run (default: all)")
    parser.add_argument("--skip", help="comma-separated stages to skip")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    stages = args.stages.split(",") if args.stages else list(ALL_STAGES)
    skip = set(args.skip.split(",")) if args.skip else set()
    stages = [s.strip() for s in stages if s.strip() not in skip]

    print(f"Running stages: {stages}")
    for name in stages:
        t0 = time.perf_counter()
        print(f"\n{'=' * 60}\n▶  {name.upper()}\n{'=' * 60}")
        _run_stage(name, cfg)
        print(f"✓  {name} done in {time.perf_counter() - t0:.1f}s")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
