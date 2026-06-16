# ML Experiments

A hands-on log of machine-learning and LLM experiments. Each folder is a
self-contained project with its own README, code, and reproducible setup.

## Projects

| Project | What it explores | Stack |
|---|---|---|
| [llm-distillation](llm-distillation/) | Distilling a large teacher LLM into a small 0.5B student for Amazon ESCI search-relevance classification — teacher labelling, QLoRA fine-tuning, evaluation vs gold labels | Transformers · PEFT/TRL · QLoRA · OpenAI API |
| [grpo-query-gen](grpo-query-gen/) | GRPO (RL) fine-tuning of an LLM to generate search queries for retrieval / candidate-generation on H&M fashion data — reward = retrieval recall of the next basket | TRL (GRPO) · PEFT · sentence-transformers |

## How it's organized

Every project is independent — `cd <project>/` and follow its README to run it.
Shared theme: small, focused experiments to learn modern ML/LLM techniques end-to-end.

---
*More experiments added as I work through them.*
