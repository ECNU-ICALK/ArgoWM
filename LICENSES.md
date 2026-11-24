# License scope

**Built with Qwen.** The paired AgroWM code and model releases have
component-specific license terms. The Apache-2.0 license for AgroWM's original code does not remove the
restrictions on the Rooted Renderer weights inherited from Qwen Image 2.1.

| Component | Applicable terms |
|---|---|
| Original AgroWM code and documentation | [Apache License 2.0](LICENSE) |
| Rice and maize PhenoClock LoRA adapters | Apache-2.0; the [Qwen3.8-27B upstream license](licenses/Qwen3.8-27B-LICENSE.txt) is retained |
| Rice and maize Rooted Renderer weights and their base architecture | [Qwen Research License Agreement](licenses/Qwen-Image-2.1-LICENSE.txt), including its non-commercial research/evaluation restriction |
| Other dependencies and data | Their respective upstream licenses; see [THIRD_PARTY.md](THIRD_PARTY.md) |

The Rooted Renderers are full fine-tuned derivatives of Qwen Image 2.1. Their
upstream license permits redistribution subject to its conditions; commercial use
requires a separate license from the upstream licensor. They must not be treated
as unrestricted Apache-2.0 model weights.

The model-card metadata therefore uses a custom combined license entry pointing
to this file. It does not offer a choice of license for the Renderer.

## Attribution and modifications

- PhenoClock: Qwen3.8-27B, Copyright 2026 Alibaba Cloud. AgroWM contributes
  crop-specific LoRA adapters trained to predict structured Growth Scripts.
- Rooted Renderer: Qwen Image 2.1, Copyright (c) 2026 Hangzhou Tongyi Laboratory
  Technology Co., Ltd. AgroWM fine-tunes the diffusion transformer on source
  images and teacher Growth Scripts. Text-encoder and VAE weights are not included.
- The two Renderer files named `diffusion_pytorch_model.safetensors` contain these
  fine-tuning modifications. They are not the original base-model parameters.
- In the adapter configurations, the original private machine path is replaced
  by the public base-model identifier. No weight tensor is altered for packaging.

Required upstream notices are retained in [NOTICE](NOTICE), with copies alongside
the component files in the model repository. Upstream license texts are included unmodified under
`licenses/`.
