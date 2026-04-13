from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from dataio import projectio
from tracking.types import LineageRecord, SegmentationSource, TrackingCheckpoint, TrackingConfig


class ProjectIOTests(unittest.TestCase):
    def test_extract_frame_index_supports_current_filename_patterns(self) -> None:
        names = [
            "t000.tif",
            "man_seg001.tif",
            "mask002.tif",
            "t-003.tif",
            "example_labels-4.tif",
        ]
        indices = [projectio.extract_frame_index(Path(name)) for name in names]
        self.assertEqual(indices, [0, 1, 2, 3, 4])

    def test_discover_segmentation_sources_finds_builtin_and_external_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            extra_seg_root = tmp / "Segmentations"
            self._write_sequence(dataset_root / "02", "t", 2)
            self._write_sequence(dataset_root / "02_ST" / "SEG", "man_seg", 2)
            self._write_sequence(dataset_root / "02_ERR_SEG", "mask", 2)
            self._write_sequence(extra_seg_root / "embedseg", "t-", 2)
            self._write_sequence(extra_seg_root / "stardist", "example_labels-", 2)

            sources = projectio.discover_segmentation_sources(
                dataset_root=dataset_root,
                sequence="02",
                extra_seg_root=extra_seg_root,
                include_external=True,
            )

            self.assertEqual(set(sources), {"embedseg", "err_seg", "st", "stardist"})
            self.assertEqual(sources["st"].frame_count, 2)
            self.assertFalse(sources["embedseg"].training_capable)

    def test_load_source_frame_objects_rejects_frame_count_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            self._write_sequence(dataset_root / "02", "t", 2)
            self._write_sequence(dataset_root / "02_ST" / "SEG", "man_seg", 1)

            raw_frames = projectio.load_raw_sequence(dataset_root, "02")
            source = projectio.discover_segmentation_sources(dataset_root, "02")["st"]
            with self.assertRaises(ValueError):
                projectio.load_source_frame_objects(source, raw_frames)

    def test_load_source_frame_objects_splits_disconnected_source_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            raw_dir = dataset_root / "02"
            raw_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(raw_dir / "t000.tif", np.ones((4, 4), dtype=np.uint16))

            source_dir = tmp / "Segmentations" / "embedseg"
            source_dir.mkdir(parents=True, exist_ok=True)
            disconnected = np.zeros((4, 4), dtype=np.uint16)
            disconnected[0, 0] = 5
            disconnected[3, 3] = 5
            tifffile.imwrite(source_dir / "mask000.tif", disconnected)

            raw_frames = projectio.load_raw_sequence(dataset_root, "02")
            source = SegmentationSource(
                name="embedseg",
                sequence="02",
                frame_paths=(source_dir / "mask000.tif",),
                frame_count=1,
                shape=(4, 4),
                training_capable=False,
            )

            frames = projectio.load_source_frame_objects(source, raw_frames)

            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].object_count, 2)
            self.assertEqual(projectio.disconnected_label_components(frames[0].label_image), ())
            self.assertEqual(sorted(frames[0].areas), [1, 1])

    def test_write_tracking_outputs_uses_ctc_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            tracked_masks = np.stack(
                [
                    np.array([[0, 1], [1, 0]], dtype=np.uint16),
                    np.array([[0, 0], [2, 2]], dtype=np.uint16),
                ]
            )
            lineage_rows = (
                LineageRecord(track_id=1, begin=0, end=0, parent=0),
                LineageRecord(track_id=2, begin=1, end=1, parent=0),
            )

            mask_paths, lineage_path = projectio.write_tracking_outputs(output_dir, tracked_masks, lineage_rows)

            self.assertEqual([path.name for path in mask_paths], ["mask000.tif", "mask001.tif"])
            self.assertEqual(lineage_path.name, "res_track.txt")
            loaded = tifffile.imread(mask_paths[0])
            self.assertEqual(loaded.dtype, np.uint16)
            self.assertEqual(tuple(loaded.shape), (2, 2))

    def test_tracking_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint = TrackingCheckpoint(
                version=1,
                dataset_root="/tmp/demo",
                track_sequence="02",
                seg_source="stardist",
                selected_sources=("stardist",),
                frame_count=5,
                frame_shape=(4, 4),
                completed_frame=2,
                next_track_id=8,
                max_distance=50.0,
                segmentation_reward=-105.0,
                lineage_state={1: (0, 2, 0), 2: (1, 2, 1)},
            )

            projectio.write_tracking_checkpoint(output_dir, checkpoint)
            loaded = projectio.load_tracking_checkpoint(output_dir)

            self.assertEqual(loaded, checkpoint)

    def test_load_saved_tracking_solution_rebuilds_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            raw_frames = np.stack(
                [
                    np.full((4, 4), 5, dtype=np.uint16),
                    np.full((4, 4), 7, dtype=np.uint16),
                ]
            )
            tracked_masks = np.stack(
                [
                    np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16),
                    np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16),
                ]
            )
            projectio.write_tracking_outputs(
                output_dir,
                tracked_masks,
                (LineageRecord(track_id=1, begin=0, end=1, parent=0),),
            )
            projectio.write_tracking_checkpoint(
                output_dir,
                TrackingCheckpoint(
                    version=1,
                    dataset_root="/tmp/demo",
                    track_sequence="02",
                    seg_source="stardist",
                    selected_sources=("stardist",),
                    frame_count=2,
                    frame_shape=(4, 4),
                    completed_frame=1,
                    next_track_id=2,
                    max_distance=50.0,
                    segmentation_reward=-105.0,
                    lineage_state={1: (0, 1, 0)},
                ),
            )

            solution = projectio.load_saved_tracking_solution("stardist", output_dir, raw_frames)

            self.assertEqual(solution.source_name, "stardist")
            self.assertEqual(solution.frames[0].frame_index, 0)
            self.assertEqual(solution.frames[1].frame_index, 1)
            self.assertEqual(solution.frames[0].raw_label_ids, (1,))
            self.assertEqual(len(solution.lineage_rows), 1)

    def test_load_saved_tracking_solution_rejects_disconnected_saved_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            raw_frames = np.stack([np.ones((4, 4), dtype=np.uint16)])
            invalid_mask = np.zeros((4, 4), dtype=np.uint16)
            invalid_mask[0, 0] = 1
            invalid_mask[3, 3] = 1
            tifffile.imwrite(output_dir / "mask000.tif", invalid_mask)
            projectio.write_lineage_rows(output_dir, (LineageRecord(track_id=1, begin=0, end=0, parent=0),))
            projectio.write_tracking_checkpoint(
                output_dir,
                TrackingCheckpoint(
                    version=1,
                    dataset_root="/tmp/demo",
                    track_sequence="02",
                    seg_source="stardist",
                    selected_sources=("stardist",),
                    frame_count=1,
                    frame_shape=(4, 4),
                    completed_frame=0,
                    next_track_id=2,
                    max_distance=50.0,
                    segmentation_reward=-105.0,
                    lineage_state={1: (0, 0, 0)},
                ),
            )

            with self.assertRaisesRegex(ValueError, "disconnected label ids"):
                projectio.load_saved_tracking_solution("stardist", output_dir, raw_frames)

    def test_load_gt_frame_objects_rejects_disconnected_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            self._write_sequence(dataset_root / "02", "t", 1)
            gt_dir = dataset_root / "02_GT" / "TRA"
            gt_dir.mkdir(parents=True, exist_ok=True)
            invalid_mask = np.zeros((4, 4), dtype=np.uint16)
            invalid_mask[0, 0] = 1
            invalid_mask[3, 3] = 1
            tifffile.imwrite(gt_dir / "man_track000.tif", invalid_mask)

            raw_frames = projectio.load_raw_sequence(dataset_root, "02")

            with self.assertRaisesRegex(ValueError, "disconnected label ids"):
                projectio.load_gt_frame_objects(dataset_root, "02", raw_frames)

    def test_load_gt_tracking_reference_ignores_disconnected_track_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            self._write_sequence(dataset_root / "02", "t", 2)
            gt_dir = dataset_root / "02_GT" / "TRA"
            gt_dir.mkdir(parents=True, exist_ok=True)

            invalid_mask = np.zeros((4, 4), dtype=np.uint16)
            invalid_mask[0, 0] = 1
            invalid_mask[3, 3] = 1
            tifffile.imwrite(gt_dir / "man_track000.tif", invalid_mask)

            valid_mask = np.zeros((4, 4), dtype=np.uint16)
            valid_mask[1:3, 1:3] = 2
            tifffile.imwrite(gt_dir / "man_track001.tif", valid_mask)

            (gt_dir / "man_track.txt").write_text("1 0 0 0\n2 1 1 1\n", encoding="utf-8")

            raw_frames = projectio.load_raw_sequence(dataset_root, "02")
            frames, lineage_records = projectio.load_gt_tracking_reference(
                dataset_root,
                "02",
                raw_frames,
                ignore_disconnected_tracks=True,
            )

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].object_count, 0)
            self.assertEqual(frames[1].raw_label_ids, (2,))
            self.assertEqual(sorted(lineage_records), [2])
            self.assertEqual(lineage_records[2], LineageRecord(track_id=2, begin=1, end=1, parent=0))

    def test_load_lineage_records_falls_back_to_res_track_for_toy_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = tmp / "Fluo-N2DL-HeLa"
            gt_dir = dataset_root / "02_GT" / "TRA"
            gt_dir.mkdir(parents=True, exist_ok=True)
            (gt_dir / "res_track.txt").write_text("7 0 1 0\n9 2 3 7\n", encoding="utf-8")

            records = projectio.load_lineage_records(dataset_root, "02")

            self.assertEqual(sorted(records), [7, 9])
            self.assertEqual(records[7], LineageRecord(track_id=7, begin=0, end=1, parent=0))
            self.assertEqual(records[9], LineageRecord(track_id=9, begin=2, end=3, parent=7))

    def test_toy_mask007_outputs_keep_track_ids_connected(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        mask_paths = (
            repo_root / "outputs" / "Fluo-N2DL-HeLa" / "toy" / "embedseg" / "mask007.tif",
            repo_root / "outputs" / "Fluo-N2DL-HeLa" / "toy" / "stardist" / "mask007.tif",
            repo_root / "outputs" / "Fluo-N2DL-HeLa" / "toy" / "consensus_embedseg_stardist_joint" / "optimized_joint" / "mask007.tif",
            repo_root / "outputs" / "Fluo-N2DL-HeLa" / "toy" / "consensus_embedseg_stardist_two_stage" / "optimized_two_stage" / "mask007.tif",
        )
        if not all(path.exists() for path in mask_paths):
            self.skipTest("Toy regression outputs are not available in this workspace.")

        for path in mask_paths:
            image = tifffile.imread(path)
            self.assertEqual(projectio.disconnected_label_components(image), (), str(path))

    def _write_sequence(self, directory: Path, prefix: str, frame_count: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for frame_index in range(frame_count):
            image = np.zeros((4, 4), dtype=np.uint16)
            image[1:3, 1:3] = frame_index + 1
            tifffile.imwrite(directory / f"{prefix}{frame_index:03d}.tif", image)


if __name__ == "__main__":
    unittest.main()
