# Reported results and comparison scope

Values below are transcribed from Tables 1–2 and the long-horizon analysis in
*Can World Models Predict Rice Growth? Toward Agricultural World Models*.
They are reported manuscript results, not measurements rerun for this repository.
The machine-readable transcription is [paper_results.json](paper_results.json).

## Rice: selected comparisons

DINO is higher-is-better. Area, Height, Density and Yellow are image-proxy errors
in percentage points, lower-is-better. Stage and Event R@1 are percentages,
higher-is-better. This table selects relevant comparisons rather than reproducing
all external baselines in the manuscript.

| Method | DINO | Area | Height | Density | Yellow | Stage R@1 | Event R@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT Image 2 | 0.709 | 5.59 | 11.49 | 6.31 | 24.40 | 35.3 | 12.8 |
| GPT Image 2.5 | 0.721 | 6.92 | 12.57 | 6.70 | 29.17 | 34.8 | 10.8 |
| Qwen Image 2.1 | 0.693 | 7.85 | 18.28 | 6.83 | 32.31 | 24.0 | 1.1 |
| Source copy | 0.702 | 7.17 | 19.06 | 6.24 | 29.33 | 23.2 | 0.0 |
| Qwen Image 2.1 FT / Simple | 0.834 | 3.73 | 7.83 | 5.33 | 11.43 | 47.4 | 22.5 |
| **AgroWM** | **0.860** | **3.51** | **7.43** | **4.98** | **10.00** | **54.6** | **24.8** |

At 9–20 weeks, AgroWM achieves **46.4% Stage R@1**, compared with **35.7%** for
Simple FT: a **10.7 percentage-point** difference.

## Maize

| Method | DINO ↑ | Area error (pp) ↓ | Height error (pp) ↓ | Stage R@1 (%) ↑ |
|---|---:|---:|---:|---:|
| Source copy | 0.522 | 3.62 | 33.02 | 0.0 |
| GPT Image 2 | 0.727 | 2.63 | 16.64 | 21.1 |
| Qwen Image 2.1 FT / Simple | 0.764 | 1.97 | 11.12 | 30.1 |
| **AgroWM** | **0.795** | **1.73** | **10.61** | **30.9** |

Maize uses independently trained models and exact day intervals. It is a second
crop experiment, not zero-shot transfer of the rice weights. Its Stage R@1 gain
over Simple FT is 0.8 points.

## What the metrics measure

- **DINO:** DINOv2 Base feature cosine similarity, using the pinned preprocessing.
- **Area / Height / Density / Yellow:** visible plant proxies in the prepared
  image plane. Rice Height is front-view-only; use the dedicated geometry code.
  Maize Area/Height come from the maize phenotype evaluator.
- **Stage R@1:** the nearest retrieved image has the target stage, within a
  plant/view-specific gallery. It does not score the Planner's stage text.
- **Event R@1:** the retrieved image matches the target image/event. Maize's
  one-image-per-stage setup makes stage and event retrieval coincide.

The nominal validation sets contain 2,000 rice and 1,600 maize queries. A nominal
query count is not automatically every metric's denominator: rice Height uses
front views only, retrieval applies gallery eligibility, and phenotype scores
retain metric-specific validity gates. Use the emitted per-metric counts.

Rice retrieval requires at least five canonical images per plant/view; maize
requires two. Galleries are constructed from canonical images present in the
scored evaluation rows. Selecting only a few successful examples changes the
retrieval task. Preserve the complete aligned output set, native image geometry,
original stage labels and frozen per-query seeds.

## Public execution conditions

| Condition | Script | Image generator | Source image |
|---|---|---|---|
| AgroWM / Ours | Predicted by PhenoClock | Trained text Rooted Renderer | Yes |
| Simple FT | None | Independently trained Qwen Image 2.1 | Yes |
| Tutor reference | Target-informed teacher | Same Renderer as Ours | Yes |
| Base + predicted script | Predicted by PhenoClock | Original Qwen Image 2.1 | Yes |
| Current Qwen zero-shot | None | Original Qwen Image 2.1 | Yes |
| Source copy | None | No generation | Copied |

The public **Base + predicted script** command is source-conditioned. The paper's
historical **w/o Rooted Renderer** row used a script-only base generator without
the source photograph, so the command above must not be assigned that row's score.
The current zero-shot command uses the full current Simple prompt; archived rice
baseline prompts can differ. Maize zero-shot qualitative examples are not a full
1,600-query benchmark. Preserve these distinctions when reporting reruns.

External baseline outputs can be evaluated through the result-ledger interface in
[RUNNING.md](RUNNING.md). Closed-model APIs and the full training stacks of external
methods are not included. Exact reproduction requires the original prepared data,
model versions, prompts, seeds and all relevant preprocessing choices.
