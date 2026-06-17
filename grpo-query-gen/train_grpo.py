"""GRPO fine-tuning: train an LLM to generate search queries whose retrieved
candidates cover a customer's next basket.

Run on a GPU (Colab T4). GRPO samples `num_generations` completions per prompt,
so it's generation-heavy — keep the model/group small and tune down on OOM.
"""
import inspect
import os

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from src.data import ID_CACHE, TYPE_CACHE, VEC_CACHE
from src.reward import Retriever, make_reward_fn

MODEL = os.environ.get("GRPO_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")  # 1.5B for quality if it fits


def main():
    dataset = load_dataset("json", data_files="data/dataset.jsonl", split="train")
    retriever = Retriever(VEC_CACHE, ID_CACHE, TYPE_CACHE)
    reward_func = make_reward_fn(retriever, k=20, match="type")  # denser than exact-id

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             task_type="CAUSAL_LM", target_modules="all-linear")

    cfg_kwargs = dict(
        output_dir="artifacts/grpo-query-gen",
        num_generations=8,               # group size G (relative-advantage baseline)
        per_device_train_batch_size=8,   # must be a multiple of num_generations
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=64,        # queries are short
        learning_rate=1e-6,
        beta=0.04,                       # KL penalty to the frozen reference
        temperature=1.0,
        num_train_epochs=1,
        logging_steps=5,
        fp16=True,                       # T4 has no bf16
        report_to="none",
    )
    # GRPOConfig args drift across trl versions — pass only what this version accepts.
    valid = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = [k for k in cfg_kwargs if k not in valid]
    if dropped:
        print("note: GRPOConfig ignoring unsupported args:", dropped)
    cfg = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    trainer = GRPOTrainer(model=MODEL, reward_funcs=reward_func, args=cfg,
                          train_dataset=dataset, peft_config=peft_config)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print("saved ->", cfg.output_dir)


if __name__ == "__main__":
    main()
