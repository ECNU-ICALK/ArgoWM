# Mainline experiments

For inference using the published Ours weights, start with
[CHECKPOINTS.md](CHECKPOINTS.md) and continue with the evaluation section here.
The training workflow below remains available for reproducing Ours and Simple FT.

Run from the repository root. First complete INSTALL.md and DATA.md, then generate
`configs/local/`. Examples below use the default `data/` and `runs/` layout. If you
changed `data_root` or `run_root`, use those locations in explicit path arguments.

Choose the crop once. Both species are trained independently from their base
models; maize is not zero-shot transfer from rice.

```bash
crop=rice                         # or maize
planner_python=.venv-planner/bin/python
renderer_python=.venv-renderer/bin/python
```

## 1. Train PhenoClock

```bash
# Preview and validate the prepared package first.
"$planner_python" scripts/run_planner_sft.py \
  --config "configs/local/$crop/planner_train.json" --print-json

# Launch the published recipe.
"$planner_python" scripts/run_planner_sft.py \
  --config "configs/local/$crop/planner_train.json" --launch
```

Rice: 2 GPUs, FSDP2, 2,500 updates. Maize: 8 GPUs, ZeRO-3, 800 updates.
Both use global batch 16, two epochs, LoRA rank 32/alpha 64 and grouped
stage/metric/text losses 0.1/0.4/0.5. The generated config preserves these settings.
The public Planner recipe retains one saved checkpoint (`save_total_limit=1`);
this changes retention, not training.

For a bounded runtime smoke check, use `--max-steps 1 --output-dir runs/smoke-planner`
with the prepared package and `--launch`. This still loads the full model and
requires the configured hardware. It is not a paper training run.

After training, copy the actual final adapter directory into the corresponding
`rice_planner_adapter` or `maize_planner_adapter` entry of `configs/local.json`.
Generate fresh configs, for example:

```bash
python scripts/configure.py --settings configs/local.json --output configs/local-inference
```

Use `configs/local-inference/` in the inference commands below. Do not choose an
adapter from another crop or a historical unweighted run.

## 2. Train the Rooted Renderer and Simple FT

Both models start from the original Qwen Image 2.1 base. Training the text Renderer
uses teacher scripts; inference for Ours uses scripts predicted by PhenoClock.

```bash
# Run sequentially to avoid two jobs competing for the same eight GPUs.
for variant in teacher_cot simple; do
  "$renderer_python" scripts/run_executor_diffsynth.py \
    --config "configs/local/$crop/executor_$variant.json" \
    --variant "$variant" --phase cache --launch
  "$renderer_python" scripts/run_executor_diffsynth.py \
    --config "configs/local/$crop/executor_$variant.json" \
    --variant "$variant" --phase train --launch
done
```

Without `--launch`, the launcher validates the input package and prints commands.
Even previews require the base model layout and the DiffSynth checkout. Cache
creation encodes text/images; training consumes the validated cached tensors.
Never reuse a cache generated from a different prompt or geometry. The trainer
saves `epoch-0.safetensors` and final `epoch-1.safetensors`; inference selects the
latter. Completed run directories are protected against accidental overwriting.

## 3. Predict scripts with PhenoClock

```bash
# One replica over two GPUs; increase groups only when memory/hardware permit.
"$planner_python" scripts/launch_planner_inference.py \
  --config "configs/local-inference/$crop/planner_infer.json" \
  --output-dir "runs/$crop/planner_inference" \
  --batch-size 1 --gpu-groups '0,1'
```

For the original 8-GPU inference topology, `--gpu-groups '0,1;2,3;4,5;6,7'` creates
four replicas. Each group's memory must fit the Planner. Rice's template allows
80 GiB per GPU; adjust `max_memory_per_gpu` to the actual hardware when needed.

Outputs include the per-query ledger, `executor_prompts.jsonl`, `rejected.jsonl`
and a summary. Inputs contain source photographs only. The exporter checks IDs,
source-image identity, prompt format and adapter provenance. Invalid scripts are
reported; there is no Tutor fallback. Full image inference requires full canonical
query coverage. If any scripts are rejected, inspect and resolve the run rather
than silently evaluating a smaller set as the full experiment.

## 4. Generate future images

```bash
for variant in predicted_cot simple; do
  "$renderer_python" scripts/run_executor_inference.py \
    --config "configs/local-inference/$crop/executor_infer.json" \
    --variant "$variant"
done
```

Default inference uses one GPU, 20 steps, BF16 and the frozen per-query seeds.
Change `renderer_infer_gpus` before materializing configs to shard across GPUs.
For a bounded image-generation check, append `--row-indices 0`; use a distinct
`--output-root` for that smoke run. Selected examples do not support full-gallery
Stage R@1 claims.

Optional reference/control conditions:

```bash
# Tutor: target-informed reference, sharing Ours' Renderer checkpoint.
"$renderer_python" scripts/run_executor_inference.py \
  --config "configs/local-inference/$crop/executor_infer.json" --variant teacher_cot

# Same predicted script with the original, untrained base image generator.
"$renderer_python" scripts/run_executor_inference.py \
  --config "configs/local-inference/$crop/executor_base.json" --variant predicted_cot --pretrained

# Qwen Image 2.1 zero-shot with the current crop's Simple prompt.
"$renderer_python" scripts/run_executor_inference.py \
  --config "configs/local-inference/$crop/executor_base.json" --variant simple --pretrained
```

These zero-shot commands use the current full Simple template. Some historical
rice paper baselines used older prompts; do not call this rerun an exact reproduction
of their archived numbers. Maize's supplementary zero-shot examples were not a
complete 1,600-query baseline.

## 5. Evaluate generated images

Pixel/foreground/color metrics first, with the same full validation set for each
method. Source copy is prepared without duplicating image bytes.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$renderer_python" scripts/evaluate_image_results.py \
  --dataset-manifest "data/$crop/evaluation/manifest.jsonl" \
  --result "ours=runs/$crop/executor_inference/predicted_cot" \
  --result "simple=runs/$crop/executor_inference/simple" \
  --result "source_copy=data/$crop/evaluation/source_copy" \
  --output-dir "runs/$crop/evaluation/pixels" --workers 8
```

Add Tutor or other conditions through additional `--result NAME=PATH` arguments.
Use native geometry. An explicitly selected `--resize-predictions` protocol is
required for mismatched external output sizes; raw external images should be kept.

DINOv2 Base and retrieval:

```bash
min_gallery=5
if [ "$crop" = maize ]; then min_gallery=2; fi
"$renderer_python" scripts/augment_dino_recall_metrics.py \
  --dataset-manifest "data/$crop/evaluation/manifest.jsonl" \
  --evaluation-dir "runs/$crop/evaluation/pixels" \
  --output-dir "runs/$crop/evaluation/dino" \
  --dino-checkpoint models/dinov2-base/dinov2_vitb14_pretrain.pth \
  --devices 0 --batch-size 64 \
  --min-gallery-images-per-plant-view "$min_gallery" --merge-evaluation
```

`--merge-evaluation` augments the current pixel outputs in place. The gallery is
built from canonical images present in the **evaluation rows**, not independently
from every row of `--dataset-manifest`. Use complete aligned output sets; evaluating
a selected subset changes the retrieval task and denominators.

Rice geometry (front-view Height and foreground Density):

```bash
"$renderer_python" scripts/augment_geometry_metrics.py \
  --dataset-manifest data/rice/evaluation/manifest.jsonl \
  --evaluation-dir runs/rice/evaluation/pixels \
  --output-dir runs/rice/evaluation/geometry --workers 8
```

Maize phenotypes use the separate crop-specific stage:

```bash
"$renderer_python" scripts/evaluate_maize_phenotypes.py \
  --dataset-manifest data/maize/evaluation/manifest.jsonl \
  --result ours=runs/maize/executor_inference/predicted_cot \
  --result simple=runs/maize/executor_inference/simple \
  --result source_copy=data/maize/evaluation/source_copy \
  --output-dir runs/maize/evaluation/phenotypes --workers 8
```

Read `summary.json`, `per_pair_results.jsonl`, alignment reports and per-metric
valid counts. For maize, use DINO/Stage from the augmented pixel summary and
Area/Height from the phenotype summary. For rice, Height/Density come from geometry.

External GPT / other baseline outputs can use the same evaluator: write a JSONL
ledger with `pair_id`, `output_path` and `status: "completed"`, retain raw images,
and add it via `--result`. This package does not contain private API-provider
settings or run commercial APIs automatically.
