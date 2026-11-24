# Experimental protocol

## Inputs and supervision

| Property | Rice | Maize |
|---|---|---|
| Query | Source image, accession, front/top view, rounded weeks | Source image, plant ID, side view, exact days |
| Horizon | `max(1, (actual_interval_days + 3) // 7)` | Positive integer days |
| Planner output | Direct `<GROWTH_COT>`; native thinking disabled | Same wrapper and thinking policy |
| Stage vocabulary in script | S1–S8 | S1–S9 |
| Native image geometry | 672×896 front; 896×672 top | 768×1280 side |
| Training | Independently from base models | Independently from base models |

Source/target stage labels and target images are evaluation/training metadata,
not query conditions. The system prompt contains the general stage vocabulary;
it does not reveal the queried plant's annotated stage. A Growth Script has three
ordered parts: **Stage reasoning**, **Growth plan**, **Target phenotype**. Numeric
slots depend on crop and view. See the exact files in
`protocols/planner_v3_direct_metric_named_weeks/`,
`protocols/qwen_image21_executor_v9_v3_metric_named_weeks/` and
`protocols/maize_four_way_v1/`. Small synthetic examples are under `tests/fixtures/`;
they are format examples, not measured plant observations.

## Training configuration

| Setting | Rice Planner | Maize Planner | Renderer / Simple FT |
|---|---|---|---|
| Base | Qwen3.8-27B | Qwen3.8-27B | Qwen Image 2.1 |
| Trainable parameters | LoRA, all-linear; vision/aligner frozen | Same | Full DiT; text encoder/VAE frozen via caches |
| LoRA | rank 32, alpha 64, dropout 0.05 | Same | Not used |
| Epochs / global batch | 2 / 16 | 2 / 16 | 2 / 16 |
| Devices / per-device batch | 2 / 8 | 8 / 2 | 8 / 2 |
| Accumulation | 1 | 1 | 1 |
| Distribution | FSDP2 | ZeRO-3 | ZeRO-3 |
| Learning rate | 1e-4 | 1e-4 | 1e-5 |
| Schedule | Cosine, minimum 1e-6 | Same | 10% warmup, 60% stable, 30% cosine |
| Warmup | 3% | 3% | 10%, starts at zero |
| Final LR | 1e-6 | 1e-6 | 1e-6 |
| Weight decay | 0 | 0 | 0.01 |
| Max text length | 4096 | 4096 | Model pipeline |
| Max pixels | 602112 | 983040 | 602112 rice; 983040 maize |
| Seed | 42 (data seed 42) | Same | 20261030 |
| Precision | BF16 | BF16 | BF16 |

Grouped Planner loss is `0.1 L_stage + 0.4 L_metric + 0.5 L_text`. Token losses are
averaged within fields, then fields and examples are averaged according to the
versioned implementation. Prompt/image/padding tokens are excluded. This is not
equivalent to weighting the aggregate ordinary next-token loss.

Image inference uses 20 steps, original per-query seeds and native image geometry.
The shared text Renderer checkpoint serves both Planner predictions and the
Tutor reference. Simple FT is a separate model initialized from the base.

## Evaluation

- DINO cosine uses DINOv2 Base. timm preprocessing is bicubic resize/518×518 center
  crop with ImageNet normalization; use the pinned implementation.
- Stage R@1 is nearest-image retrieval within the same plant/view sequence. It is
  not the Planner's stage classification accuracy. Rice Stage and Event recall
  differ; maize has one image per plant/view/stage, so they coincide.
- Rice gallery eligibility is at least 5 canonical images per plant/view. Maize
  requires at least 2. Canonical images are collected from scored evaluation rows.
- Rice Height is `100 * (H - y_min) / H`, valid for front view only. Top-view Height
  is unavailable. Do not replace it with the pixel evaluator's legacy extent.
- Maize Height is computed by the dedicated phenotype script using the registered
  crop measurement. It is an image proxy in percentage points, not centimeters.
- Area, Density and yellow/brown measures are image-plane proxies. Empty masks or
  insufficient color coverage remain unavailable; never replace missing values
  with zero. Report each metric's valid denominator.
- Preserve near-white normalization, alpha compositing and original stage labels.
  Paired uncertainty analyses should cluster by plant and use valid intersections.
