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


def _build_dataset(cfg: Config, tokenizer):
    """Render each labeled example into a single training text using the
    student's own chat template (prompt + target completion)."""
    from datasets import Dataset

    rows = list(read_jsonl(Path(cfg.paths.processed_dir) / "train_labeled.jsonl"))
    if not rows:
        raise FileNotFoundError(
            "No labeled data found. Run scripts/02_generate_teacher.py first."
        )

    def render(row: dict) -> dict:
        messages = row["messages"] + [{"role": "assistant", "content": row["completion"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return Dataset.from_list([render(r) for r in rows])


def train(cfg: Config) -> Path:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)
    t = cfg.train
    out_dir = Path(cfg.paths.artifacts_dir) / t.output_subdir

    tokenizer = AutoTokenizer.from_pretrained(t.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if t.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        t.student_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=t.epochs,
        learning_rate=t.learning_rate,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        max_seq_length=t.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("Saved student adapter -> %s", out_dir)
    return out_dir
