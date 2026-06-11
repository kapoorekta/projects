# Running on Google Colab (free T4)

The free T4 GPU handles this project's QLoRA training comfortably. Copy each
block below into its own Colab cell.

> **First:** Runtime ▸ Change runtime type ▸ Hardware accelerator = **T4 GPU**.

---

### Cell 1 — confirm you actually have a GPU

```python
!nvidia-smi
```

If this errors, the runtime isn't on GPU — fix the runtime type above first.

### Cell 2 — clone the repo and install

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>/llm-distillation
!pip install -q -e ".[teacher,train,dev]"
```

### Cell 3 — set your API key (for the teacher labels)

Use Colab's secret manager (🔑 icon in the left sidebar) to add `OPENAI_API_KEY`,
then:

```python
from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
# os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")  # if using Anthropic
```

### Cell 4 — (optional) persist outputs to Google Drive

Colab wipes local disk when the session ends. Mount Drive and point artifacts there
so your trained adapter survives:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/llm-distillation/artifacts
!ln -sfn /content/drive/MyDrive/llm-distillation/artifacts artifacts
```

### Cell 5 — run the whole pipeline

```python
!python scripts/run_pipeline.py --config configs/config.yaml
```

Or run it in pieces (e.g. get a no-training baseline first):

```python
# Baseline: data + teacher labels + eval of the UNTRAINED student
!python scripts/run_pipeline.py --stages data,teacher,eval

# Then train and re-evaluate to measure the lift
!python scripts/run_pipeline.py --stages train,eval
```

### Cell 6 — see the results

```python
import json
print(json.dumps(json.load(open("artifacts/student-qlora/metrics.json")), indent=2))
```

---

## Tips

- **Save before you disconnect.** If you skipped the Drive step, download
  `artifacts/student-qlora/` (or push the adapter to the Hugging Face Hub) before
  the session ends — free runtimes reset their disk.
- **Idle timeout.** Free Colab disconnects after ~90 min idle and caps total
  session length. Keep the tab active during training.
- **Out of memory?** In `configs/config.yaml` lower `per_device_batch_size`
  (try 4 or 2) and/or `max_seq_len` (try 768), or raise `grad_accum_steps` to
  keep the effective batch size.
- **Too slow / out of credits?** Shrink `data.n_train` (e.g. 500) for a quick
  smoke test of the full loop before scaling up.
