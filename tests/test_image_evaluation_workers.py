from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.io import read_jsonl, write_jsonl


class EvaluationWorkerTests(unittest.TestCase):
    def test_parallel_matches_serial_with_missing_and_invalid_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.png", root / "target.png"
            for path, box in ((source, (8, 8, 18, 35)), (target, (5, 4, 24, 38))):
                image = Image.new("RGB", (32, 48), "white")
                ImageDraw.Draw(image).rectangle(box, fill=(40, 120, 50))
                image.save(path)
            invalid = root / "invalid.png"
            invalid.write_bytes(b"not an image")
            dataset = [{"row_index": i, "pair_id": f"pair-{i}", "pair_group_id": f"group-{i}",
                        "view": "side0", "split": "val", "source_image_path": str(source),
                        "target_image_path": str(target)} for i in range(6)]
            metadata = root / "dataset.jsonl"
            write_jsonl(metadata, dataset)
            complete = root / "complete.jsonl"
            partial = root / "partial.jsonl"
            write_jsonl(complete, [{**row, "output_path": str(source), "status": "completed"} for row in dataset])
            write_jsonl(partial, [{**row, "output_path": str(invalid if row["row_index"] == 2 else target),
                                  "status": "completed"} for row in dataset if row["row_index"] != 1])
            outputs = []
            for workers in (1, 2):
                output = root / f"workers-{workers}"
                subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_image_results.py"),
                                "--dataset-manifest", str(metadata), "--result", f"complete={complete}",
                                "--result", f"partial={partial}", "--output-dir", str(output),
                                "--workers", str(workers)], check=True, capture_output=True, text=True,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
                outputs.append(output)
            serial, parallel = (read_jsonl(path / "per_pair_results.jsonl") for path in outputs)
            self.assertEqual(serial, parallel)
            self.assertEqual([row["row_index"] for row in serial], [i for i in range(6) for _ in range(2)])
            summaries = [json.loads((path / "summary.json").read_text())["models"] for path in outputs]
            self.assertEqual(*summaries)
            self.assertEqual(summaries[0]["complete"]["completed"], 6)
            self.assertEqual(summaries[0]["partial"]["completed"], 4)
            self.assertEqual(summaries[0]["partial"]["missing_prediction"], 1)
            self.assertEqual(summaries[0]["partial"]["invalid"], 1)
            required = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_image_results.py"),
                "--dataset-manifest", str(metadata), "--result", f"partial={partial}",
                "--output-dir", str(root / "required"), "--require-complete"],
                capture_output=True, text=True,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            self.assertEqual(required.returncode, 1)


if __name__ == "__main__":
    unittest.main()
