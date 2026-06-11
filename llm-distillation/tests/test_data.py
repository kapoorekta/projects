"""Unit tests that run without network, a GPU, or API keys."""
from llm_distill.config import load_config
from llm_distill.data import _to_example
from llm_distill.prompts import LABELS, build_messages, target_completion
from llm_distill.teacher import _parse


def test_config_loads_defaults():
    cfg = load_config("configs/config.yaml")
    assert cfg.seed == 42
    assert cfg.data.dataset_id
    assert cfg.train.lora_r > 0


def test_to_example_maps_label():
    row = {
        "esci_label": "Substitute",
        "query": "running shoes",
        "product_title": "Trail Sneakers",
        "product_locale": "us",
        "example_id": 7,
    }
    ex = _to_example(row, 0, max_chars=100)
    assert ex["gold_label"] == "S"
    assert ex["query"] == "running shoes"
    assert ex["id"] == "7"


def test_to_example_rejects_bad_label():
    assert _to_example({"esci_label": "???", "query": "x"}, 0, 100) is None
    assert _to_example({"esci_label": "Exact", "query": ""}, 0, 100) is None


def test_build_messages_shape():
    msgs = build_messages("phone case", {"product_title": "Silicone Case"})
    assert msgs[0]["role"] == "system"
    assert "phone case" in msgs[1]["content"]


def test_parse_handles_json_and_noise():
    label, rationale = _parse('Sure: {"label": "E", "rationale": "exact match"} done')
    assert label == "E"
    assert rationale == "exact match"
    assert _parse("not json")[0] is None


def test_target_completion_roundtrips():
    import json

    out = json.loads(target_completion("C", "complementary item"))
    assert out["label"] == "C" and out["label"] in LABELS
