<div align="center">

# AgroWM

**Can World Models Predict Rice Growth? Toward Agricultural World Models**

[![Paper](https://img.shields.io/badge/arXiv-2611.26145-B31B1B)](https://arxiv.org/abs/2611.26145)
[![Code](https://img.shields.io/badge/GitHub-Code-181717)](https://github.com/ECNU-ICALK/ArgoWM)
[![Model](https://img.shields.io/badge/Hugging_Face-Models-FFD21E)](https://huggingface.co/ecnu-icalk/ArgoWM)

**English** · [简体中文](README_zh-CN.md)

Built with Qwen

</div>

**AgroWM** predicts how the same plant will look days to months into the future from a single photograph. It first reasons about the plant's development, then generates its future appearance. We study this task on rice and maize.

> **Reason about development. Then render growth.**

## How it works

[![AgroWM framework: PhenoClock predicts a Growth Script, and the Rooted Renderer edits the source photograph.](assets/method-overview.png)](assets/method-overview.pdf)

*Figure 3 from the paper. Click the figure for the vector PDF.*

- **PhenoClock** infers the current growth stage, predicts the future stage, and writes a **Growth Script** with targets for plant height, area, color, and organs.
- **Rooted Renderer** follows the script to edit the source photograph while preserving the plant's identity and viewpoint.

PhenoClock is built on **Qwen3.8-27B**; the Rooted Renderer is built on **Qwen Image 2.1**. See the [Growth Script examples](examples/README.md) for the intermediate format.

## Main results

Results reported in the paper, compared with the same image backbone fine-tuned without Growth Scripts (Simple FT):

| Crop | Method | DINO ↑ | Area error ↓ | Height error ↓ | Stage R@1 (%) ↑ |
|---|---|---:|---:|---:|---:|
| Rice | Simple FT | 0.834 | 3.73 | 7.83 | 47.4 |
| Rice | **AgroWM** | **0.860** | **3.51** | **7.43** | **54.6** |
| Maize | Simple FT | 0.764 | 1.97 | 11.12 | 30.1 |
| Maize | **AgroWM** | **0.795** | **1.73** | **10.61** | **30.9** |

At **9–20 weeks** on rice, AgroWM reaches **46.4% Stage R@1**, compared with **35.7%** for Simple FT. Rice and maize models are trained independently.

Area and Height are image-proxy errors in percentage points. Stage R@1 measures whether a forecast retrieves the target stage from the same plant's observed image sequence. See [full results and evaluation details](docs/RESULTS.md).

## Quick start

This repository contains the training, inference, and evaluation code. The paired [Hugging Face repository](https://huggingface.co/ecnu-icalk/ArgoWM) provides the final rice and maize models.

Clone the code, then download the model files once they are available on Hugging Face:

```bash
git clone https://github.com/ECNU-ICALK/ArgoWM.git
cd ArgoWM

python -m pip install -U huggingface_hub
hf download ecnu-icalk/ArgoWM   --include 'checkpoints/**' --local-dir .
```

If you already have the full Hugging Face repository, use that directory directly. The four released parameter files total about **30 GB**.

For inference, use **Linux, Python 3.12 and NVIDIA GPUs**:

1. [Install the environments](docs/INSTALL.md) and obtain the two required base models.
2. [Prepare your data](docs/DATA.md). The current workflow reads prepared image-pair manifests.
3. [Configure and run AgroWM](docs/CHECKPOINTS.md#run-the-existing-benchmark-workflow) with the released weights.

To reproduce training or evaluate predictions, follow the [training and evaluation guide](docs/RUNNING.md). Checkpoints include **AgroWM only**; base models, datasets, and baseline weights are obtained separately.

## Documentation

| Guide | Contents |
|---|---|
| [Models and inference](docs/CHECKPOINTS.md) | Components, loading, inference commands, and intended use |
| [Training and evaluation](docs/RUNNING.md) | Rice and maize experiment workflows |
| [Prompts](protocols/README.md) · [Growth Scripts](examples/README.md) | Prompt templates and script formats |
| [Experimental settings](docs/PROTOCOL.md) | Training configuration and metric definitions |
| [Code and model releases](docs/REPOSITORIES.md) | Matching versions, downloads, and checksums |

The experiments use white-background rice and maize images. Field scenes and other crops require further validation. Questions and bug reports are welcome through [GitHub Issues](https://github.com/ECNU-ICALK/ArgoWM/issues).

## Citation

If you use AgroWM in your research, please cite our [paper](https://arxiv.org/abs/2611.26145):

```bibtex
@misc{ji2026agrowm,
  title         = {Can World Models Predict Rice Growth? Toward Agricultural World Models},
  author        = {Wenkai Ji and Zeteng Li and Yinkun Guo and Jie Zhou and Liang He and Zhenong Jin},
  year          = {2026},
  eprint        = {2611.26145},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.26145}
}
```

## License

Code, documentation, and PhenoClock adapters use **[Apache-2.0](LICENSE)**. Rooted Renderer weights follow the **[Qwen Research License](LICENSES.md)** for non-commercial research and evaluation. See [LICENSES.md](LICENSES.md) for component-specific terms and [NOTICE](NOTICE) for attribution.
