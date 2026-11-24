# Released AgroWM checkpoints

The paired Hugging Face model release includes the final **Ours** model for rice and maize. Each crop has
two components: a PhenoClock LoRA adapter and a Rooted Renderer diffusion
transformer. The two crops were trained independently from their original bases.

| Crop | Component | Final training point | Parameter file | Size |
|---|---|---|---|---:|
| Rice | PhenoClock | Step 2,500; epoch 2 | `checkpoints/rice/phenoclock/adapter_model.safetensors` | 933,974,032 bytes |
| Rice | Rooted Renderer | Epoch 2 (`epoch-1` in training) | `checkpoints/rice/rooted_renderer/diffusion_pytorch_model.safetensors` | 14,230,284,584 bytes |
| Maize | PhenoClock | Step 800; epoch 2 | `checkpoints/maize/phenoclock/adapter_model.safetensors` | 467,063,552 bytes |
| Maize | Rooted Renderer | Epoch 2 (`epoch-1` in training) | `checkpoints/maize/rooted_renderer/diffusion_pytorch_model.safetensors` | 14,230,284,584 bytes |

The four weight files total **29,861,606,752 bytes** (29.86 GB; 27.81 GiB).
Rice's saved adapter uses FP32; maize's uses BF16. Both have the same rank-32 /
alpha-64 architecture. The release preserves each original weight file without
casting, merging, or quantization. Renderer weights are BF16 full diffusion
transformers, not LoRA adapters and not complete image-generation pipelines.

The release includes no Simple FT or other baseline weights, intermediate
checkpoints, optimizer states, or training logs. Tutor is a target-informed
reference using the same Renderer as Ours; it does not require a separate model.

## Use the matching code release

Both repositories identify this release as **`agrowm-mainline-2026-11-24`** in
[configs/model_release.json](../configs/model_release.json). The code repository
contains no weight files; the model repository contains the matching code snapshot
and `checkpoints/`. See [REPOSITORIES.md](REPOSITORIES.md) for how to connect the
two local checkouts or download the model files into a code checkout.

## Files and integrity

```text
checkpoints/
  manifest.json
  SHA256SUMS
  rice/
    phenoclock/
      adapter_config.json
      adapter_model.safetensors
      additional_config.json
    rooted_renderer/
      config.json
      diffusion_pytorch_model.safetensors
  maize/
    phenoclock/
      adapter_config.json
      adapter_model.safetensors
      additional_config.json
    rooted_renderer/
      config.json
      diffusion_pytorch_model.safetensors
```

The model repository's `checkpoints/manifest.json` records component identities,
tensor dtypes, file sizes, and SHA-256 digests. Both repositories carry the same
weight identities in [configs/model_release.json](../configs/model_release.json).
After placing the model repository's `checkpoints/` directory in your checkout,
verify its `checkpoints/SHA256SUMS` from the checkout root:

```bash
# Linux
sha256sum -c checkpoints/SHA256SUMS

# macOS
shasum -a 256 -c checkpoints/SHA256SUMS
```

Obtain the actual large files when downloading this repository. A Git LFS pointer
is a small text file, not a usable checkpoint. Use `hf download` or run
`git lfs pull` in a Git clone.

The adapter configurations identify `Qwen/Qwen3.8-27B` instead of the original
machine-local base path. All other adapter fields are preserved. Each Renderer
configuration matches the architecture used by the original base transformer.

## Required base models

- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): the frozen Planner
  backbone, tokenizer, image processor and template support.
- [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1): the image pipeline's
  frozen text encoder, VAE, scheduler and other supporting files.

Download these separately and set their local directories in the settings file.
The supplied inference scripts use local model paths and offline loading. The
repository root is a composite model release, not a single Transformers model.
Install the separate Planner and Renderer stacks in [INSTALL.md](INSTALL.md).

## Run the existing benchmark workflow

The released parameters let you skip training. First place or link the model
repository's `checkpoints/` directory in your code checkout as described in
[REPOSITORIES.md](REPOSITORIES.md); the model checkout already has it. Prepare the data according to
[DATA.md](DATA.md); the current workflow consumes prepared validation manifests.
Its source-only request builder strips target photos, labels and teacher answers
from model inputs.

These commands use the default `run_root=runs`. If you change `run_root`, also
replace the Planner `--output-dir` below with
`<your run_root>/$crop/planner_inference`; the Renderer reads the script manifest
from that configured location. Explicit data-preparation paths must likewise
match your `data_root`.

```bash
cp configs/released.example.json configs/local.released.json
# Edit the base-model, data, Python-environment and output paths.
python scripts/configure.py \
  --settings configs/local.released.json --output configs/local-released

crop=rice  # or maize
.venv-planner/bin/python scripts/launch_planner_inference.py \
  --config "configs/local-released/$crop/planner_infer.json" \
  --output-dir "runs/$crop/planner_inference" \
  --batch-size 1 --gpu-groups '0,1'

.venv-renderer/bin/python scripts/run_executor_inference.py \
  --config "configs/local-released/$crop/executor_infer.json" \
  --variant predicted_cot
```

Choose a GPU group that fits the Planner and adjust its memory limits for the
hardware. The released settings select both the crop's Planner adapter and its
Renderer file. `scripts/configure.py` writes that Renderer path into the Ours/Tutor
inference configuration; Simple FT keeps its independent training-output path.
The optional `--checkpoint` flag remains available for an explicit override.
Keep crop identities aligned: do not combine a rice adapter with a maize Renderer.

Continue with the evaluation commands in [RUNNING.md](RUNNING.md). Invalid scripts
are recorded rather than replaced by target-informed supervision. Reproducing
paper metrics also requires the original prepared inputs, seeds and full gallery.

## Load the Renderer component directly

For integration into another application, the Renderer directories are compatible
with the pinned Diffusers `QwenImage21Transformer2DModel` class:

```python
import torch
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

crop = "rice"
transformer = QwenImage21Transformer2DModel.from_pretrained(
    f"checkpoints/{crop}/rooted_renderer",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe = QwenImage21Pipeline.from_pretrained(
    "models/Qwen-Image-2.1",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")
```

This loads the image component only. Use the source image, crop-specific prompt,
predicted Growth Script, native geometry, seed, and inference settings defined by
the mainline workflow when generating a forecast. A generic text-to-image prompt
does not reproduce AgroWM.

## Intended use and limitations

AgroWM supports research on developmental reasoning, source-conditioned crop image prediction, structured supervision, and phenotype-aware evaluation.

- Evidence is limited to the prepared rice and maize distributions, including white-background plant photographs. Field images, new crops and unseen cultivars require separate validation.
- The horizon describes elapsed time. Weather, treatment, soil and management records are not explicit model inputs.
- A forecast is a generated appearance, not an observed future. Traits are image-plane proxies rather than calibrated physical measurements or yield predictions.
- Growth-balanced pair selection, preprocessing and gallery construction affect the evaluation. A selected example is not a substitute for full validation.
- Errors in predicted stages and numeric targets can propagate to image generation. Tutor results use privileged future information and are a reference, not a deployable fallback.
- Exact historical comparisons require their original prompts, model versions, seeds and outputs. Closed API models may change.

## License

**Built with Qwen.** PhenoClock adapters use [Apache-2.0](../LICENSE).
Rooted Renderer weights use the [Qwen Research License](../licenses/Qwen-Image-2.1-LICENSE.txt),
including its non-commercial research/evaluation restriction. Commercial use of
the Renderer requires a separate upstream license. See [LICENSES.md](../LICENSES.md)
for the component-specific scope and [NOTICE](../NOTICE) for attribution.
