# Third-party software and models

**Built with Qwen.** AgroWM code, documentation, and PhenoClock adapters use
[Apache-2.0](LICENSE). Rooted Renderer weights use the
[Qwen Research License](licenses/Qwen-Image-2.1-LICENSE.txt), which restricts use to
non-commercial research/evaluation. See [LICENSES.md](LICENSES.md) and [NOTICE](NOTICE).
External software, data and model parameters retain their own licenses; the
AgroWM license does not replace their terms.

| Dependency | Reference | Use |
|---|---|---|
| Qwen3.8-27B | https://huggingface.co/Qwen/Qwen3.8-27B | PhenoClock base model |
| Qwen Image 2.1 | https://huggingface.co/Qwen/Qwen-Image-2.1 | Rooted Renderer and image baselines |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio, commit `60b709c4dab8eeabce06ee9bc168328ed4c780e8` | Renderer training |
| Diffusers | https://github.com/huggingface/diffusers, commit `578c9b2c6636ab2424a0e56186268b83623656b2` | Image inference; use the full pinned revision in requirements/renderer.txt |
| ms-swift | https://github.com/modelscope/ms-swift, version 4.6.0 | Planner training and inference |
| Transformers | https://github.com/huggingface/transformers | Model and tokenizer interfaces |
| DINOv2 / timm | https://github.com/facebookresearch/dinov2 and https://github.com/huggingface/pytorch-image-models | Image evaluation |

Base-model files, third-party repositories and datasets are not vendored here.
Obtain them through their original distribution channels and retain the relevant
notices. WGSR and WGSM images belong to the Open Plant Image Archive dataset
families; follow the source datasets' access and redistribution terms.

`assets/method-overview.pdf` is the original Figure 3 extracted from the
manuscript, with a PNG preview. It has not been redrawn.
`assets/growth-script-templates.png` reproduces the project's Growth Script format
figure. Text files under `examples/` are synthetic format demonstrations, not
released dataset observations or measured model outputs.
