"""Prompt construction and the label space.

Centralizing prompts here means the teacher, the student's training targets,
and eval all speak the exact same format — a common source of silent bugs.
"""
from __future__ import annotations

LABELS = ["E", "S", "C", "I"]

LABEL_NAMES = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
}

# Maps the dataset's raw `esci_label` strings to our short codes.
RAW_TO_CODE = {
    "Exact": "E",
    "Substitute": "S",
    "Complement": "C",
    "Irrelevant": "I",
    # some releases use single letters already
    "E": "E", "S": "S", "C": "C", "I": "I",
}

SYSTEM_PROMPT = (
    "You are a product-search relevance expert. Given a shopping query and a "
    "product, classify how relevant the product is to the query using the ESCI scheme:\n"
    "  E (Exact): the product fully matches the query intent.\n"
    "  S (Substitute): not exact, but a reasonable alternative the shopper might accept.\n"
    "  C (Complement): used together with the queried item, but is not it.\n"
    "  I (Irrelevant): unrelated to the query.\n"
    'Respond ONLY with a compact JSON object: {"label": "E|S|C|I", "rationale": "<one sentence>"}.'
)


def build_user_prompt(query: str, product: dict, max_chars: int = 1500) -> str:
    """Render a (query, product) pair into the user turn."""
    def field(key: str) -> str:
        val = (product.get(key) or "").strip()
        return val[:max_chars]

    parts = [f"Query: {query}", f"Product title: {field('product_title')}"]
    for key, label in [
        ("product_brand", "Brand"),
        ("product_color", "Color"),
        ("product_bullet_point", "Bullets"),
        ("product_description", "Description"),
    ]:
        val = field(key)
        if val:
            parts.append(f"{label}: {val}")
    parts.append('\nReturn the JSON now.')
    return "\n".join(parts)


def build_messages(query: str, product: dict, max_chars: int = 1500) -> list[dict]:
    """Chat-format messages used by both the teacher API and student training."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, product, max_chars)},
    ]


def target_completion(label: str, rationale: str) -> str:
    """The assistant turn the student is trained to produce."""
    import json

    return json.dumps({"label": label, "rationale": rationale}, ensure_ascii=False)
