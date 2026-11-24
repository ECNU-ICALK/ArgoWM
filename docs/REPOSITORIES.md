# Matching code and model repositories

Release **`agrowm-mainline-2026-11-24`** has two distributions:

| Distribution | Contents | Local checkpoint handling |
|---|---|---|
| [GitHub code repository](https://github.com/ECNU-ICALK/ArgoWM) | Training, inference, evaluation, prompts, recipes, examples and shared documentation | Weights are obtained separately and ignored by Git |
| [Hugging Face model repository](https://huggingface.co/ecnu-icalk/ArgoWM) | The same code snapshot plus the four final Ours parameter files | `checkpoints/` is included; parameter files use Git LFS |

Both contain the same [model release catalog](../configs/model_release.json).
It records the release ID, component paths, base models, crop-specific protocols,
weight sizes and SHA-256 digests. The model repository's
`checkpoints/manifest.json` carries the same release ID and component identities.
Do not combine the rice Planner with the maize Renderer or substitute Simple FT
for the released text Renderer.

The matching trees are `src/`, `scripts/`, `configs/`, `protocols/`,
`requirements/`, `tests/`, `.github/`, `docs/`, `assets/`, `examples/`, and
`licenses/`, together with `pyproject.toml` and the license/notice files.
The two READMEs describe the distribution they accompany; `.gitignore` and
`.gitattributes` differ to keep weights out of the code repository.

## Use the two local folders together

If you have both distributions next to one another, run the following **from the
code repository root**. Replace the model-folder path if your layout differs:

```bash
ln -s ../AgroWM-HuggingFace/checkpoints checkpoints
```

The model directory must contain the complete `checkpoints/` tree before linking.
Create this link only when the code checkout has no `checkpoints` entry. If it
already contains weights, inspect that directory before making changes. The link
does not copy the 29.86 GB of parameters, and the code repository ignores it.

Alternatively, keep the models outside the code checkout and edit these four
fields in your own settings JSON to their actual locations:

```json
{
  "rice_planner_adapter": "/path/to/model-release/checkpoints/rice/phenoclock",
  "rice_renderer_checkpoint": "/path/to/model-release/checkpoints/rice/rooted_renderer/diffusion_pytorch_model.safetensors",
  "maize_planner_adapter": "/path/to/model-release/checkpoints/maize/phenoclock",
  "maize_renderer_checkpoint": "/path/to/model-release/checkpoints/maize/rooted_renderer/diffusion_pytorch_model.safetensors"
}
```

Use `configs/released.example.json` as the starting point so base-model, data and
environment paths are also configured. Settings resolve relative paths against
the code checkout root and accept absolute paths.

## Obtain model files from the Hub

After the model files are published to [ecnu-icalk/ArgoWM](https://huggingface.co/ecnu-icalk/ArgoWM),
run the command from the code checkout root, using an environment with the
Hugging Face CLI installed:

```bash
hf download ecnu-icalk/ArgoWM \
  --repo-type model --include 'checkpoints/**' --local-dir .
```

This downloads only the checkpoint tree into the code checkout and preserves its
code files. It needs approximately 30 GB of free space. The model files require
the [component-specific licenses](../LICENSES.md); downloading the code alone
does not supply the base models or dataset.

## Configure and run either distribution

Once the checkpoints are available, the same commands work in either checkout:

These commands use the default `run_root=runs`. If you change `run_root`, also
replace the Planner `--output-dir` below with
`<your run_root>/$crop/planner_inference`; the Renderer reads the script manifest
from that configured location. Explicit data-preparation paths must likewise
match your `data_root`.

```bash
cp configs/released.example.json configs/local.released.json
# Edit base-model, data, environment and output locations.
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

Prepare the environments and data first; see [INSTALL.md](INSTALL.md) and
[DATA.md](DATA.md). The generated Planner and Renderer configurations point to
the matched crop components automatically. Simple FT and the untrained base
controls retain their independent paths. To train your own models, use
`configs/local.example.json`; its optional checkpoint fields default to `null`,
preserving the original training-output layout.

## Verify the downloaded parameters

After copying, linking, or downloading the complete checkpoint tree, run from the
checkout root:

```bash
# Linux
sha256sum -c checkpoints/SHA256SUMS

# macOS
shasum -a 256 -c checkpoints/SHA256SUMS
```

Match `release_id` in `configs/model_release.json` and
`checkpoints/manifest.json`, and retain the corresponding protocol files. Changes
to those prompts, crop units or preprocessing can change results even when the
weight hashes match.
