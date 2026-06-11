"""Typed config loaded from a single YAML file.

Dataclasses give us autocomplete + a clear contract for what each stage needs,
while keeping the experiment knobs in one human-editable file.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


@dataclass
class DataConfig:
    dataset_id: str = "tasksource/esci"
    locale: str = "us"
    use_small_version: bool = True
    n_train: int = 2000
    n_eval: int = 500
    max_product_chars: int = 1500
    streaming: bool = True       # stream from HF instead of downloading the full set
    shuffle_buffer: int = 10000  # streaming shuffle buffer size


@dataclass
class TeacherConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_concurrency: int = 8
    max_retries: int = 4
    keep_gold_label: bool = True


@dataclass
class TrainConfig:
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_subdir: str = "student-qlora"
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_batch_size: int = 8
    grad_accum_steps: int = 2
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True


@dataclass
class EvalConfig:
    compare: list[str] = field(default_factory=lambda: ["student", "teacher"])
    max_new_tokens: int = 128


@dataclass
class Config:
    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(cls)}
    extra = set(data) - known
    if extra:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(extra)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        seed=raw.get("seed", 42),
        paths=_build(Paths, raw.get("paths", {})),
        data=_build(DataConfig, raw.get("data", {})),
        teacher=_build(TeacherConfig, raw.get("teacher", {})),
        train=_build(TrainConfig, raw.get("train", {})),
        eval=_build(EvalConfig, raw.get("eval", {})),
    )
