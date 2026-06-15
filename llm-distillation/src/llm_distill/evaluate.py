"""Score the student (and optionally the teacher) against gold ESCI labels.

Stage input:  data/interim/eval.jsonl  (+ trained student adapter)
Stage output: artifacts/<output_subdir>/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .prompts import LABELS, build_messages
from .teacher import _call, _parse
from .utils import get_logger, read_jsonl

log = get_logger(__name__)


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    # Predictions that failed to parse become "?" and count as wrong.
    paired = [(t, p or "?") for t, p in zip(y_true, y_pred)]
    yt = [t for t, _ in paired]
    yp = [p for _, p in paired]
    return {
        "n": len(yt),
        "accuracy": round(accuracy_score(yt, yp), 4),
        "macro_f1": round(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0), 4),
        "parse_failures": sum(1 for p in y_pred if p not in LABELS),
        "report": classification_report(yt, yp, labels=LABELS, zero_division=0, output_dict=True),
    }


def _student_predict(cfg: Config, examples: list[dict]) -> list[str]:
    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(cfg.paths.artifacts_dir) / cfg.train.output_subdir
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.train.student_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    preds: list[str] = []
    for ex in tqdm(examples, desc="student eval"):
        messages = build_messages(ex["query"], ex["product"], cfg.data.max_product_chars)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=cfg.eval.max_new_tokens, do_sample=False
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        label, _ = _parse(text)
        preds.append(label)
    return preds


def _teacher_predict(cfg: Config, examples: list[dict]) -> list[str]:
    from tqdm import tqdm

    preds: list[str] = []
    for ex in tqdm(examples, desc="teacher eval"):
        messages = build_messages(ex["query"], ex["product"], cfg.data.max_product_chars)
        try:
            label, _ = _parse(_call(cfg, messages))
        except RuntimeError:
            label = None
        preds.append(label)
    return preds


def evaluate(cfg: Config) -> dict:
    examples = list(read_jsonl(Path(cfg.paths.interim_dir) / "eval.jsonl"))
    gold = [ex["gold_label"] for ex in examples]
    results: dict[str, dict] = {}

    if "student" in cfg.eval.compare:
        log.info("Evaluating student on %d examples", len(examples))
        results["student"] = _metrics(gold, _student_predict(cfg, examples))
    if "teacher" in cfg.eval.compare:
        log.info("Evaluating teacher on %d examples", len(examples))
        results["teacher"] = _metrics(gold, _teacher_predict(cfg, examples))

    out_path = Path(cfg.paths.artifacts_dir) / cfg.train.output_subdir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    for name, m in results.items():
        log.info("%-8s acc=%.3f  macro_f1=%.3f  (n=%d)", name, m["accuracy"], m["macro_f1"], m["n"])
    log.info("Wrote metrics -> %s", out_path)
    return results
