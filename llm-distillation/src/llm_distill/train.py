"""Fine-tune the small student on the teacher-labeled data with QLoRA.

Stage input:  data/processed/train_labeled.jsonl
Stage output: artifacts/<output_subdir>/  (LoRA adapter + tokenizer)

Heavy deps (torch/transformers/peft/trl/bitsandbytes) are imported lazily so
the rest of the project works on a machine without a GPU.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .utils import get_logger, read_jsonl, set_seed

log = get_logger(__name__)


def _balance(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    """Resample to `per_class` examples per gold label: downsample classes that
    have more, oversample (with replacement) classes that have fewer."""
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["gold_label"]].append(r)

    out: list[dict] = []
    for items in by_label.values():
        rng.shuffle(items)
        if len(items) >= per_class:
            out.extend(items[:per_class])
        else:
            out.extend(items)
            out.extend(rng.choices(items, k=per_class - len(items)))
    rng.shuffle(out)
    return out


def _build_dataset(cfg: Config, tokenizer):
    """Render each labeled example into a single training text using the
    student's own chat template (prompt + target completion)."""
    from collections import Counter

    from datasets import Dataset

    rows = list(read_jsonl(Path(cfg.paths.processed_dir) / "train_labeled.jsonl"))
    if not rows:
        raise FileNotFoundError(
            "No labeled data found. Run scripts/02_generate_teacher.py first."
        )

    if cfg.train.balance_classes:
        rows = _balance(rows, cfg.train.balance_per_class, cfg.seed)
        log.info("Balanced classes -> %s (%d total)",
                 dict(sorted(Counter(r["gold_label"] for r in rows).items())), len(rows))

    def render(row: dict) -> dict:
        messages = row["messages"] + [{"role": "assistant", "content": row["completion"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return Dataset.from_list([render(r) for r in rows])


def train(cfg: Config) -> Path:
    import inspect

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)
    t = cfg.train
    out_dir = Path(cfg.paths.artifacts_dir) / t.output_subdir

    # T4 (Turing) has no native bf16 — pick the dtype the GPU actually supports.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(t.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if t.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        t.student_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    peft_config = LoraConfig(
        r=t.lora_r,
        lora_alpha=t.lora_alpha,
        lora_dropout=t.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    dataset = _build_dataset(cfg, tokenizer)
    log.info("Training student %s on %d examples", t.student_model, len(dataset))

    # trl renames things across versions — build kwargs against the installed signature.
    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=t.epochs,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        dataset_text_field="text",
        report_to="none",
    )
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    # max_seq_length (older trl) was renamed to max_length (newer trl).
    sft_kwargs["max_seq_length" if "max_seq_length" in sft_params else "max_length"] = t.max_seq_len
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(
        model=model, args=sft_config, train_dataset=dataset, peft_config=peft_config
    )
    # tokenizer= (older trl) was renamed to processing_class= (newer trl).
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("Saved student adapter -> %s", out_dir)
    return out_dir
