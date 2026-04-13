from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import tifffile

import build_consensus_toy_dataset as toy
from tracking.types import LineageRecord


class ImageReadTests(unittest.TestCase):
    def test_read_image_falls_back_to_pillow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "frame.tif"
            original = np.arange(16, dtype=np.uint16).reshape(4, 4)
            Image.fromarray(original).save(path)

            with mock.patch("build_consensus_toy_dataset.tifffile.imread", side_effect=ValueError("requires the 'imagecodecs' package")):
                loaded = toy.read_image(path)

        np.testing.assert_array_equal(loaded, original)


class SourceEventTests(unittest.TestCase):
    def test_extract_source_consensus_events_detects_common_only_and_split_merge_cases(self) -> None:
        left = np.zeros((3, 6, 6), dtype=np.uint16)
        right = np.zeros((3, 6, 6), dtype=np.uint16)

        left[0, 0:2, 0:2] = 1
        right[0, 0:2, 0:2] = 10
        left[0, 3:5, 0:2] = 2
        right[0, 3:5, 3:5] = 11

        left[1, 1:5, 1:5] = 3
        right[1, 1:3, 1:5] = 20
        right[1, 3:5, 1:5] = 21

        left[2, 1:3, 1:5] = 4
        left[2, 3:5, 1:5] = 5
        right[2, 1:5, 1:5] = 30

        events = toy.extract_source_consensus_events(
            left,
            right,
            left_name="embedseg",
            right_name="stardist",
            agreement_iou_threshold=0.8,
            overlap_iou_threshold=0.1,
        )

        kinds = {event.kind for event in events}
        self.assertIn("common", kinds)
        self.assertIn("embedseg_only", kinds)
        self.assertIn("stardist_only", kinds)
        self.assertIn("split_disagreement", kinds)
        self.assertIn("merge_disagreement", kinds)


class CropAssessmentTests(unittest.TestCase):
    def test_assess_gt_crop_rejects_temporal_gaps(self) -> None:
        gt_masks = np.zeros((3, 6, 6), dtype=np.uint16)
        gt_masks[0, 1:3, 1:3] = 1
        gt_masks[2, 1:3, 1:3] = 1
        crop = toy.CropWindow(start_frame=0, frame_count=3, top=0, left=0, height=4, width=4)

        assessment = toy.assess_gt_crop(gt_masks, crop)

        self.assertFalse(assessment.valid)
        self.assertEqual(assessment.gapped_track_ids, (1,))

    def test_rebuild_cropped_lineage_rows_detaches_missing_parent(self) -> None:
        gt_masks = np.zeros((3, 6, 6), dtype=np.uint16)
        gt_masks[1, 0:2, 0:2] = 2
        gt_masks[2, 0:2, 0:2] = 2
        crop = toy.CropWindow(start_frame=0, frame_count=3, top=0, left=0, height=3, width=3)
        lineage_rows = {
            1: LineageRecord(track_id=1, begin=0, end=0, parent=0),
            2: LineageRecord(track_id=2, begin=1, end=2, parent=1),
        }

        rows = toy.rebuild_cropped_lineage_rows(gt_masks, crop, lineage_rows)

        self.assertEqual(rows, (LineageRecord(track_id=2, begin=1, end=2, parent=0),))

    def test_write_selection_outputs_preserves_ctc_gt_and_st_conventions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            raw_source_dir = tmp / "raw_source"
            extra_seg_output_root = tmp / "Segmentations_toy_consensus"
            raw_source_dir.mkdir(parents=True, exist_ok=True)

            raw_paths: list[Path] = []
            for frame_index in range(2):
                raw_path = raw_source_dir / f"t{frame_index:03d}.tif"
                tifffile.imwrite(raw_path, np.full((4, 4), frame_index, dtype=np.uint16))
                raw_paths.append(raw_path)

            gt_masks = np.zeros((2, 4, 4), dtype=np.uint16)
            gt_masks[:, 1:3, 1:3] = 1
            gt_seg_masks_by_frame = {
                1: np.pad(np.full((2, 2), 7, dtype=np.uint16), 1),
            }
            st_seg_masks_by_frame = {
                0: np.pad(np.full((2, 2), 3, dtype=np.uint16), ((0, 2), (0, 2))),
                1: np.pad(np.full((3, 3), 4, dtype=np.uint16), ((1, 0), (1, 0))),
            }
            source_masks = {
                "embedseg": gt_masks.copy(),
                "stardist": gt_masks.copy(),
            }
            crop = toy.CropWindow(start_frame=0, frame_count=2, top=0, left=0, height=4, width=4)
            lineage_rows = {
                1: LineageRecord(track_id=1, begin=0, end=1, parent=0),
            }

            _raw_output_dir, gt_output_root, _extra_seg_root = toy.write_selection_outputs(
                dataset_root=dataset_root,
                extra_seg_output_root=extra_seg_output_root,
                output_sequence="toy_consensus",
                crop=crop,
                raw_paths=tuple(raw_paths),
                gt_masks=gt_masks,
                gt_seg_masks_by_frame=gt_seg_masks_by_frame,
                st_seg_masks_by_frame=st_seg_masks_by_frame,
                source_masks=source_masks,
                lineage_rows=lineage_rows,
                force=True,
            )

            gt_tra_dir = gt_output_root / "TRA"
            gt_seg_dir = gt_output_root / "SEG"
            st_seg_dir = dataset_root / "toy_consensus_ST" / "SEG"
            self.assertTrue((gt_tra_dir / "man_track.txt").exists())
            self.assertEqual((gt_tra_dir / "man_track.txt").read_text(encoding="utf-8"), "1 0 1 0\n")
            self.assertEqual(sorted(path.name for path in gt_seg_dir.glob("*.tif")), ["man_seg001.tif"])
            np.testing.assert_array_equal(
                tifffile.imread(gt_seg_dir / "man_seg001.tif"),
                gt_seg_masks_by_frame[1],
            )
            self.assertEqual(sorted(path.name for path in st_seg_dir.glob("*.tif")), ["man_seg000.tif", "man_seg001.tif"])
            np.testing.assert_array_equal(tifffile.imread(st_seg_dir / "man_seg000.tif"), st_seg_masks_by_frame[0])
            np.testing.assert_array_equal(tifffile.imread(st_seg_dir / "man_seg001.tif"), st_seg_masks_by_frame[1])


if __name__ == "__main__":
    unittest.main()
