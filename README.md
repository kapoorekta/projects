# ML Experiments

A collection of end-to-end machine learning and LLM experiments exploring modern
techniques applied to real-world problems and datasets. Each is a self-contained
directory with its own README, source code, and reproducible setup.

## Projects

| Project | Description | Stack |
|---|---|---|
| [llm-distillation](llm-distillation/) | Distillation of a large teacher LLM into a 0.5B student for Amazon ESCI search-relevance classification (E/S/C/I), covering teacher labelling, QLoRA fine-tuning, and evaluation against gold labels. | Transformers · PEFT/TRL · QLoRA · OpenAI API |
| [grpo-query-gen](grpo-query-gen/) | GRPO reinforcement-learning fine-tuning of an LLM to generate search queries for retrieval-based recommendation on the H&M dataset; the reward is retrieval recall of the customer's next basket. | TRL (GRPO) · PEFT · sentence-transformers |

## Organization

Each project is independent: change into its directory and follow the project
README to reproduce it.
