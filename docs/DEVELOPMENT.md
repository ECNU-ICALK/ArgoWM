# Tests and development

Run from the repository root after installing the relevant environment and
`python -m pip install --no-deps -e .`. The package is intended to be used as a
checkout, so protocol and script paths remain available.

## CPU contract checks

```bash
python -m compileall -q src scripts tests
python -m unittest -v \
  tests.test_release_data tests.test_release_config tests.test_mainline_protocols \
  tests.test_text_executor_release \
  tests.test_evaluation_io tests.test_image_evaluation_workers \
  tests.test_image_geometry_evaluation tests.test_v9_image_metrics
```

These checks cover preparation, source-only inputs, prompt formats, configuration,
identity and resume behavior, image geometry and metric contracts. The text
inference integration test substitutes the GPU/model boundary while exercising
real preparation, manifests, worker/supervisor control flow, image I/O and ledgers.

## Complete unit suite

Use the Renderer dependencies for tests that need Torch, Accelerate, timm and
Transformers:

```bash
python -m unittest discover -s tests -v
```

No unit test starts a full training run, downloads model weights or calls a paid
API. Passing the suite does not establish GPU reproduction of the paper.

## Compatibility rules

- Preserve prompt text, schema IDs, frozen splits, row order, seeds and crop units.
- Keep the internal `wgsr_v9` namespace stable for existing experiment artifacts.
- Keep text-script numeric fields; they are distinct from numeric injection.
- Tutor is target-informed supervision/reference, never an inference fallback.
- Keep local configs, credentials, data, training states and generated results outside Git.
- The code repository ignores all local checkpoints. The paired model repository
  tracks only the four selected parameter files under `checkpoints/` with Git LFS.
- Keep the shared files of both releases byte-identical; see [REPOSITORIES.md](REPOSITORIES.md).
