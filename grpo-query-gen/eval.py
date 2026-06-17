"""Evaluate base vs GRPO: recall@k of the customer's next basket from generated
queries, on the held-out eval split. The before/after number for the run.
"""
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import ID_CACHE, TYPE_CACHE, VEC_CACHE
from src.reward import Retriever, parse_queries

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "artifacts/grpo-query-gen"
K = 20


def generate(model, tok, prompt, max_new=64) -> str:
    text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def recall(retriever, text, basket, k) -> float:
    cands = set()
    for q in parse_queries(text):
        cands.update(retriever.retrieve(q, k))
    return float(np.mean([1.0 if b in cands else 0.0 for b in basket]))


def score(model, tok, retriever, ds, k) -> float:
    return float(np.mean([recall(retriever, generate(model, tok, ex["prompt"]),
                                  ex["basket"], k) for ex in ds]))


def main():
    ds = load_dataset("json", data_files="data/eval.jsonl", split="train")
    retriever = Retriever(VEC_CACHE, ID_CACHE, TYPE_CACHE)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto").eval()
    print(f"base  recall@{K}: {score(base, tok, retriever, ds, K):.4f}  (n={len(ds)})")

    grpo = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto"),
        ADAPTER).eval()
    print(f"grpo  recall@{K}: {score(grpo, tok, retriever, ds, K):.4f}  (n={len(ds)})")


if __name__ == "__main__":
    main()
