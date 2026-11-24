# Installation

Use Linux, Python 3.12 and NVIDIA GPUs for training and inference. Data preparation
and the CPU contract tests also work on macOS. Run commands from this repository's
root. Keep Planner and Renderer environments separate: their PEFT and DeepSpeed
versions differ.

## CPU preparation and inspection

```bash
python3.12 -m venv .venv-cpu
source .venv-cpu/bin/activate
python -m pip install -r requirements/cpu.txt
python -m pip install --no-deps -e .
```

## Planner

```bash
python3.12 -m venv .venv-planner
source .venv-planner/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/planner.txt
python -m pip install -r requirements/cpu.txt
python -m pip install --no-deps -e .
```

The pinned stack uses ms-swift 4.6.0 / Transformers 5.19.0 with the `qwen3_8`
template. `causal-conv1d` and `flash-linear-attention` may require compatible wheels
or a CUDA development toolchain. Match the Python version, Torch version, CUDA
version and C++ ABI; do not substitute an arbitrary wheel. The original rice run
used 2 large-memory GPUs with FSDP2, while maize used 8 × 24 GB GPUs with ZeRO-3.

## Renderer and evaluation

```bash
python3.12 -m venv .venv-renderer
source .venv-renderer/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/renderer.txt
mkdir -p third_party
git clone https://github.com/modelscope/DiffSynth-Studio.git third_party/DiffSynth-Studio
git -C third_party/DiffSynth-Studio checkout 60b709c4dab8eeabce06ee9bc168328ed4c780e8
python -m pip install -e third_party/DiffSynth-Studio
# Reapply the experiment pins if the upstream install changed shared dependencies.
python -m pip install -r requirements/renderer.txt
python -m pip install --no-deps -e .
python -m pip check
python -c 'from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel; from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline'
```

Training checks the DiffSynth Git commit and requires a clean checkout. Inference
uses Diffusers commit `578c9b2c6636ab2424a0e56186268b83623656b2`, which provides the
Qwen Image 2.1 classes. The versions above were read from the working experiment
environments; installation on a new machine still depends on upstream package
and model availability. They are not a claim of a newly tested clean installation.

The original Renderer recipe uses 8 GPUs, batch 2 per GPU and global batch 16.
Cache creation requires substantial scratch storage; the launcher checks for at
least 150 GiB free. Public configs use disk-backed caches by default to avoid a
machine-specific `/dev/shm` layout. This changes cache placement, not the math.

## Use the released checkpoints

For inference with the paired AgroWM model release, use `configs/released.example.json`
and follow [CHECKPOINTS.md](CHECKPOINTS.md). Original base models are still required.
The settings below are the separate route for training your own models.

## Configure paths

```bash
cp configs/local.example.json configs/local.json
# Edit all model, data, environment and output locations in this file.
python scripts/configure.py --settings configs/local.json
```

Relative paths in the settings file resolve against the repository root; `~` is
expanded. Generated JSON configs contain resolved absolute paths and are ignored
by Git. The generator refuses to overwrite an existing config directory; use
`--output configs/local-v2` when changing settings, and pass that location to the
commands in RUNNING.md. Keep custom local directories out of Git as well.

The two Planner adapter fields start as `null`. After training, set each to the
actual final `checkpoint-*` directory and generate a new config directory.
The optional `rice_renderer_checkpoint` and `maize_renderer_checkpoint` fields
also default to `null`, so the training workflow selects its standard final
Renderer output. Set them when using the paired model release or another local
Renderer directory; only the Ours/Tutor path is overridden, not Simple FT or
the untrained base controls.
An absent adapter is an error for Planner inference, not a request to use the base
model. Inference GPU counts are configurable. Training topology is validated
against the published recipe; fewer-GPU training requires an explicitly revised
recipe and is not silently presented as the same experiment.
