# Data and base models

Prepare the dataset, teacher annotations and third-party base models separately,
following their access and redistribution terms. DINOv2 is also an external
evaluation dependency. The training workflow produces the local model files used
by AgroWM inference; see [RUNNING.md](RUNNING.md).

## Portable pair manifest

`prepare_data.py` consumes a JSONL file with one object per image forecast. Paths
are absolute or relative to the directory containing this JSONL file. Inputs are
already cleaned, white-background RGB photographs at the experiment resolution;
this release starts after background removal and teacher annotation.

| Field | Contract |
|---|---|
| `pair_id` | Unique string across both splits |
| `pair_group_id` | Pair grouping ID; rice front/top share one group |
| `plant_id` | Biological plant ID; no train/val overlap |
| `split` | `train` or `val`, using the frozen experimental split |
| `view` | Rice `front`/`side0` or `top`; maize `sv1`/`side0` |
| `accession_id` | Required for rice; cultivar/accession condition |
| `actual_interval_days` | Positive integer; rice prompts derive rounded weeks |
| `source_image_path`, `target_image_path` | Real paired photographs |
| `source_stage`, `target_stage` | Original evaluation labels, preserved verbatim |
| `source_stage_id`, `target_stage_id` | Optional alternative if evaluation already uses these IDs |
| `source_image_id`, `target_image_id` | Stable canonical image/event identifiers |
| `source_capture_date`, `target_capture_date` | Preserve original capture dates when available |
| `seed` | Required explicit integer; preserve the original per-query seed |
| `teacher_cot` or `teacher_cot_path` | Exact rendered `<GROWTH_COT>` teacher block |
| `horizon_bucket` | Required for maize validation: original `short`, `mid` or `long` grouping; optional for rice |

**Evaluation stage labels are distinct from script supervision.** In particular,
rice's historical gallery labels can be names such as `Tillering` or `Grain
filling`; do not replace them with the script's finer S1–S8 labels. If both label
and ID fields exist, evaluation uses `source_stage` / `target_stage`.

Teacher scripts contain target information for training supervision. They appear
only in the assistant response and Tutor condition; source-only Planner queries
are constructed from an allowlist. Maize's stage guide is part of the system
instructions, while the query does not supply the plant's true S1–S9 labels.

```bash
python scripts/prepare_data.py --crop rice --pairs /datasets/rice/pairs.jsonl \
  --output data/rice --paper-split
python scripts/prepare_data.py --crop maize --pairs /datasets/maize/pairs.jsonl \
  --output data/maize --paper-split
```

The importer validates every image, unique IDs, consistent image metadata, split
isolation, teacher format and seeds before writing. It preserves row order within
each split and refuses an existing destination. It does not resample pairs or
invent missing teacher text, labels or seeds. Omitting `--paper-split` permits small
contract tests; the published training launcher still requires the full train
count. Small examples are not paper evaluations.

| Crop | Train | Validation | Validation plants | Image size (width × height) |
|---|---:|---:|---:|---|
| Rice | 20,000 | 2,000 | 79 | Front 672×896; top 896×672 |
| Maize | 6,400 | 1,600 | 84 | Side 768×1280 |

The importer writes:

```text
data/<crop>/
  package_manifest.json
  planner/{train,val}.jsonl
  planner/package_manifest.json
  executor/{simple,teacher_cot}/{train,val}.jsonl
  executor/package_manifest.json
  evaluation/manifest.jsonl
  evaluation/seeds.jsonl
  evaluation/source_copy/results.jsonl
```

Images are not copied. Prepared files point to resolved local images. Re-run
preparation from the portable pair manifest on a new machine rather than blindly
replacing absolute paths in frozen audit files. The input and generated file
checksums are recorded. A prepared package is valid only for its corresponding
image tree. Data access and the frozen portable pair manifest must accompany the
future dataset release.

## Base-model layout

Provide a complete local Qwen3.8-27B directory (config, tokenizer, safetensors
index and all shards). Provide the complete Qwen Image 2.1 directory:

```text
models/Qwen-Image-2.1/
  model_index.json
  transformer/     # config, index and all diffusion_pytorch_model-*.safetensors
  text_encoder/    # all model-*.safetensors
  vae/diffusion_pytorch_model.safetensors
  processor/
  tokenizer/      # if supplied by the base-model package
  scheduler/
```

Keep the provider's original configuration files. Inference converts the trained
DiffSynth DiT checkpoint into a separate Diffusers loading view; it does not edit
the base weights. DINOv2 Base uses `dinov2_vitb14_pretrain.pth`, passed explicitly
to the evaluation command.

## Local training outputs

After training, set the Planner adapter path to the directory produced by your own
run; ms-swift creates a timestamped parent directory. The Simple and text Renderer
runs produce separate output directories. Ours and Tutor use the text Renderer,
while Simple uses its independently trained model. The inference commands and
local path settings are documented in [RUNNING.md](RUNNING.md).

Rebuild latent caches from the prepared metadata in a fresh package. Do not treat
leftover completion markers from deleted caches as usable tensors.
