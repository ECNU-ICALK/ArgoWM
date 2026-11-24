from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wgsr_v9.evaluation_io import pair_key_for_dataset_row  # noqa: E402


class EvaluationAlignmentTests(unittest.TestCase):
    def test_executor_request_id_aligns_to_dataset_pair_id(self) -> None:
        dataset_row = {"pair_id": "plant_a_2020-01-01_to_2020-01-08_side0"}
        executor_result = {
            "request_id": "plant_a_2020-01-01_to_2020-01-08_side0",
            "row_index": 12,
        }
        self.assertEqual(
            pair_key_for_dataset_row(dataset_row),
            pair_key_for_dataset_row(executor_result),
        )


if __name__ == "__main__":
    unittest.main()
