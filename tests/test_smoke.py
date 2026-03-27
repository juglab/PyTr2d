from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from main import run_tracking
from tracking.types import TrackingConfig


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "Fluo-N2DL-HeLa_train" / "Fluo-N2DL-HeLa"
EXTRA_SEG_ROOT = Path(__file__).resolve().parents[1] / "data" / "Fluo-N2DL-HeLa_train" / "Segmentations"
RUN_SMOKE = os.environ.get("PYTR2D_RUN_SMOKE") == "1"


@unittest.skipUnless(RUN_SMOKE, "Set PYTR2D_RUN_SMOKE=1 to run dataset smoke tests.")
class SmokeTests(unittest.TestCase):
    def test_single_source_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tracking(
                TrackingConfig(
                    dataset_root=DATASET_ROOT,
                    extra_seg_root=EXTRA_SEG_ROOT,
                    seg_source="st",
                    model_dir=Path(tmpdir) / "models",
                    output_dir=Path(tmpdir),
                )
            )
            self.assertEqual(len(result.mask_paths), 92)
            self.assertTrue(result.lineage_path.exists())

    def test_multi_source_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_tracking(
                TrackingConfig(
                    dataset_root=DATASET_ROOT,
                    extra_seg_root=EXTRA_SEG_ROOT,
                    seg_source="all",
                    model_dir=Path(tmpdir) / "models",
                    output_dir=Path(tmpdir),
                )
            )
            self.assertEqual(len(result.mask_paths), 92)
            self.assertIn("st", result.selected_sources)
            self.assertIn("stardist", result.selected_sources)


if __name__ == "__main__":
    unittest.main()
