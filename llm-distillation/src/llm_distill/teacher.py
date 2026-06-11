"""Generate teacher labels + rationales for the training examples.

Stage input:  data/interim/train.jsonl
Stage output: data/processed/train_labeled.jsonl  — adds:
    {"messages": [...], "completion": "<json>", "teacher_label": "..."}

Two modes (config: teacher.keep_gold_label):
  - True  -> keep dataset gold label, teacher only writes a rationale
             (chain-of-thought distillation).
  - False -> teacher predicts the label itself (pure response distillation).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .prompts import LABELS, build_messages, target_completion
from .utils import get_logger, read_jsonl, write_jsonl

log = get_logger(__name__)


def _parse(content: str) -> tuple[str | None, str]:
    """Best-effort parse of the model's JSON reply -> (label, rationale)."""
    try:
        start, end = content.find("{"), content.rfind("}")
        obj = json.loads(content[start : end + 1])
        label = str(obj.get("label", "")).strip().upper()[:1]
        rationale = str(obj.get("rationale", "")).strip()
        return (label if label in LABELS else None), rationale
    except (ValueError, json.JSONDecodeError):
        return None, ""


def _call_openai(model: str, messages: list[dict], temperature: float, max_retries: int) -> str:
    from openai import OpenAI

    client = OpenAI()
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - retry on transient API errors
            last_err = e
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


# Opus 4.7 / 4.8 removed the sampling params — sending temperature/top_p/top_k
# returns a 400. Only forward temperature to models that still accept it.
def _anthropic_sampling(model: str, temperature: float) -> dict:
    no_sampling = ("claude-opus-4-8", "claude-opus-4-7")
    return {} if any(model.startswith(p) for p in no_sampling) else {"temperature": temperature}


def _call_anthropic(model: str, messages: list[dict], temperature: float, max_retries: int) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    system_text = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    # Cache the constant system prompt. Only caches once the prefix clears the
    # model minimum (4096 tokens Opus/Haiku, 2048 Sonnet) — a short prompt won't.
    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client.messages.create(
                model=model, system=system, messages=user_msgs, max_tokens=512,
                **_anthropic_sampling(model, temperature),
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Anthropic call failed after {max_retries} retries: {last_err}")


def _call(cfg: Config, messages: list[dict]) -> str:
    t = cfg.teacher
    if t.provider == "openai":
        return _call_openai(t.model, messages, t.temperature, t.max_retries)
    if t.provider == "anthropic":
        return _call_anthropic(t.model, messages, t.temperature, t.max_retries)
    raise ValueError(f"Unknown teacher provider: {t.provider}")


def label_training_set(cfg: Config) -> Path:
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    src = Path(cfg.paths.interim_dir) / "train.jsonl"
    examples = list(read_jsonl(src))
    log.info("Labeling %d examples with %s/%s", len(examples), cfg.teacher.provider, cfg.teacher.model)

    def process(ex: dict) -> dict | None:
        messages = build_messages(ex["query"], ex["product"], cfg.data.max_product_chars)
        try:
            raw = _call(cfg, messages)
        except RuntimeError as e:
            log.warning("Skipping %s: %s", ex["id"], e)
            return None
        pred_label, rationale = _parse(raw)
        label = ex["gold_label"] if cfg.teacher.keep_gold_label else pred_label
        if label is None:
            return None
        return {
            **ex,
            "teacher_label": pred_label,
            "messages": messages,
            "completion": target_completion(label, rationale or "(no rationale)"),
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.teacher.max_concurrency) as pool:
        for out in tqdm(pool.map(process, examples), total=len(examples)):
            if out is not None:
                results.append(out)

    out_path = Path(cfg.paths.processed_dir) / "train_labeled.jsonl"
    n = write_jsonl(out_path, results)
    log.info("Wrote %d labeled training examples -> %s", n, out_path)
    if not cfg.teacher.keep_gold_label:
        _report_teacher_agreement(results)
    return out_path


def _report_teacher_agreement(rows: list[dict]) -> None:
    agree = sum(1 for r in rows if r.get("teacher_label") == r.get("gold_label"))
    log.info("Teacher agreement with gold: %.1f%% (%d/%d)",
             100 * agree / max(len(rows), 1), agree, len(rows))
