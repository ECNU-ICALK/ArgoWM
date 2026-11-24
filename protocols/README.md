# Mainline protocols

This directory contains the prompt templates and teacher schemas used by the
paper's rice and maize pipeline. The versioned directory names identify the
experiment contracts; each crop has one supported Planner and Renderer family.

| Directory | Role | Input and output |
|---|---|---|
| `planner_v3_direct_metric_named_weeks/` | Rice PhenoClock | Source image, accession, front/top view and rounded weeks; direct `<GROWTH_COT>` output |
| `qwen_image21_executor_v9_v3_metric_named_weeks/` | Rice Rooted Renderer and Simple baseline | Shared source-image prompt; named-metric Growth Script for script-conditioned generation |
| `qwen_image21_executor_v9_v3_metric_named_weeks/cot/` | Rice teacher annotation and script rendering | Front/top schemas, biological annotation instructions and named metric bindings |
| `maize_four_way_v1/` | Maize PhenoClock, Rooted Renderer and Simple baseline | Source image, plant ID, side view and exact days; direct `<GROWTH_COT>` output |
| `maize_sv1_v1/cot_v2_en_v2/` | Maize teacher annotation and script rendering | S1–S9 vocabulary, side-view schema, six script fields and organ evidence |

`cot_v2_en_v2` is the maize teacher schema used by the mainline pipeline. Its
version suffix does not indicate an alternative experiment. Prompt/schema bytes
and protocol IDs are preserved so existing annotations and manifests retain
their identity. Some snapshot metadata records the provenance of these selected
protocols; the public entry points use only the paths listed above.

## Shared rules within each crop

- Planner training and inference use the same prepared system/user messages.
  Inference excludes the supervised assistant message and target annotations.
- Simple FT and current zero-shot inference read the same `simple/val.jsonl`.
- Tutor and Ours share the image prompt, script suffix and output format.
  Tutor receives annotated teacher text; Ours receives the Planner prediction.
- The script-only base control reads the same predicted-script manifest as Ours.
- Both scripts contain **Stage reasoning → Growth plan → Target phenotype**
  inside `<GROWTH_COT>`, with native thinking disabled.

Rice uses `max(1, (actual_interval_days + 3) // 7)` whole weeks, front/top views
and S1–S8 script stages. Maize uses exact days, one side view and S1–S9 stages.
Crop-specific organ vocabulary and numeric formatting follow the supplied
schemas. Source/target stage labels are not query inputs.

Teacher annotation uses source and target evidence to produce supervision.
Prepare those annotations before importing the frozen pairs with
`scripts/prepare_data.py`; see [the data contract](../docs/DATA.md).
The paper's image, phenotype and stage-retrieval evaluation settings are described
in [the experimental protocol](../docs/PROTOCOL.md#evaluation), with commands in
[the running guide](../docs/RUNNING.md).
