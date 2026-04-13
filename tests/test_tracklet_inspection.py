from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from dataio import projectio
from tracking.types import LineageRecord
from viz.tracklet_inspection import load_tracklet_selection, plot_tracklet_selection


class TrackletInspectionTests(unittest.TestCase):
    def test_joint_variant_resolves_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="custom_joint",
                masks=self._joint_masks(),
                manifest_variant_names=("custom_joint",),
            )

            selection = load_tracklet_selection(
                dataset_root=dataset_root,
                track_sequence="02",
                frames=[0, 1],
                track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                source_output_dirs=source_output_dirs,
                consensus_output_dir=consensus_root,
            )

            self.assertEqual(selection.views["joint"].solution_name, "custom_joint")

    def test_joint_variant_falls_back_to_default_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )

            selection = load_tracklet_selection(
                dataset_root=dataset_root,
                track_sequence="02",
                frames=[0, 1],
                track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                source_output_dirs=source_output_dirs,
                consensus_output_dir=consensus_root,
            )

            self.assertEqual(selection.views["joint"].solution_name, "optimized_joint")

    def test_loader_rejects_out_of_range_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )

            with self.assertRaisesRegex(ValueError, "out of range"):
                load_tracklet_selection(
                    dataset_root=dataset_root,
                    track_sequence="02",
                    frames=[2],
                    track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                    source_output_dirs=source_output_dirs,
                    consensus_output_dir=consensus_root,
                )

    def test_loader_rejects_unknown_track_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )

            with self.assertRaisesRegex(ValueError, "Requested track ids \\[999\\] were not found in view 'embedseg'"):
                load_tracklet_selection(
                    dataset_root=dataset_root,
                    track_sequence="02",
                    frames=[0, 1],
                    track_ids_by_view={"embedseg": [999], "stardist": [21], "joint": [31]},
                    source_output_dirs=source_output_dirs,
                    consensus_output_dir=consensus_root,
                )

    def test_crop_bbox_uses_union_across_frames_and_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )

            selection = load_tracklet_selection(
                dataset_root=dataset_root,
                track_sequence="02",
                frames=[0, 1],
                track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                source_output_dirs=source_output_dirs,
                consensus_output_dir=consensus_root,
            )

            self.assertEqual(selection.crop_bbox, (1, 1, 7, 7))

    def test_plot_marks_missing_tracks_as_not_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )
            selection = load_tracklet_selection(
                dataset_root=dataset_root,
                track_sequence="02",
                frames=[0, 1],
                track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                source_output_dirs=source_output_dirs,
                consensus_output_dir=consensus_root,
            )

            fake_pyplot = FakePyplotModule()
            fake_matplotlib = types.ModuleType("matplotlib")
            fake_matplotlib.__path__ = []
            fake_matplotlib.pyplot = fake_pyplot
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
            ):
                figure = plot_tracklet_selection(selection, padding=0)

            self.assertIs(figure, fake_pyplot.last_figure)
            self.assertIn("not present", fake_pyplot.last_figure.axes_grid[1, 0].text_calls)
            self.assertIn("not present", fake_pyplot.last_figure.axes_grid[0, 1].text_calls)

    def test_plot_uses_rows_by_frames_and_fixed_view_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_root = self._write_raw_dataset(tmp)
            source_output_dirs = self._write_standard_source_outputs(tmp)
            consensus_root = self._write_consensus_variant(
                root=tmp,
                variant_name="optimized_joint",
                masks=self._joint_masks(),
                manifest_variant_names=None,
            )
            selection = load_tracklet_selection(
                dataset_root=dataset_root,
                track_sequence="02",
                frames=[0, 1],
                track_ids_by_view={"embedseg": [11], "stardist": [21], "joint": [31]},
                source_output_dirs=source_output_dirs,
                consensus_output_dir=consensus_root,
            )

            fake_pyplot = FakePyplotModule()
            fake_matplotlib = types.ModuleType("matplotlib")
            fake_matplotlib.__path__ = []
            fake_matplotlib.pyplot = fake_pyplot
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot},
            ):
                plot_tracklet_selection(selection, padding=0)

            figure = fake_pyplot.last_figure
            self.assertEqual(figure.axes_grid.shape, (2, 3))
            self.assertTrue(figure.axes_grid[0, 0].title.startswith("embedseg"))
            self.assertTrue(figure.axes_grid[0, 1].title.startswith("stardist"))
            self.assertTrue(figure.axes_grid[0, 2].title.startswith("joint"))
            self.assertEqual(figure.axes_grid[0, 0].ylabel, "frame 0")
            self.assertEqual(figure.axes_grid[1, 0].ylabel, "frame 1")

    def _write_raw_dataset(self, tmp: Path) -> Path:
        dataset_root = tmp / "Fluo-N2DL-HeLa"
        raw_dir = dataset_root / "02"
        raw_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(raw_dir / "t000.tif", np.arange(64, dtype=np.uint16).reshape(8, 8))
        tifffile.imwrite(raw_dir / "t001.tif", np.arange(64, dtype=np.uint16).reshape(8, 8) + 10)
        return dataset_root

    def _write_standard_source_outputs(self, root: Path) -> dict[str, Path]:
        output_root = root / "outputs"
        embedseg_dir = output_root / "embedseg"
        stardist_dir = output_root / "stardist"
        projectio.write_tracking_outputs(
            embedseg_dir,
            self._embedseg_masks(),
            (LineageRecord(track_id=11, begin=0, end=0, parent=0),),
        )
        projectio.write_tracking_outputs(
            stardist_dir,
            self._stardist_masks(),
            (LineageRecord(track_id=21, begin=1, end=1, parent=0),),
        )
        return {"embedseg": embedseg_dir, "stardist": stardist_dir}

    def _write_consensus_variant(
        self,
        root: Path,
        variant_name: str,
        masks: np.ndarray,
        manifest_variant_names: tuple[str, ...] | None,
    ) -> Path:
        consensus_root = root / "outputs" / "consensus_embedseg_stardist"
        variant_dir = consensus_root / variant_name
        projectio.write_tracking_outputs(
            variant_dir,
            masks,
            (LineageRecord(track_id=31, begin=0, end=1, parent=0),),
        )
        if manifest_variant_names is not None:
            manifest_path = consensus_root / "render_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"variant_names": list(manifest_variant_names)}), encoding="utf-8")
        return consensus_root

    def _embedseg_masks(self) -> np.ndarray:
        masks = np.zeros((2, 8, 8), dtype=np.uint16)
        masks[0, 1:3, 1:3] = 11
        return masks

    def _stardist_masks(self) -> np.ndarray:
        masks = np.zeros((2, 8, 8), dtype=np.uint16)
        masks[1, 5:7, 5:7] = 21
        return masks

    def _joint_masks(self) -> np.ndarray:
        masks = np.zeros((2, 8, 8), dtype=np.uint16)
        masks[0, 2:4, 4:6] = 31
        masks[1, 3:5, 4:6] = 31
        return masks


class FakeAxis:
    def __init__(self) -> None:
        self.images: list[tuple[np.ndarray, dict[str, object]]] = []
        self.text_calls: list[str] = []
        self.title = ""
        self.ylabel = ""
        self.xticks: list[object] | None = None
        self.yticks: list[object] | None = None
        self.transAxes = object()

    def imshow(self, image: np.ndarray, **kwargs: object) -> None:
        self.images.append((np.asarray(image), dict(kwargs)))

    def set_xticks(self, ticks: list[object]) -> None:
        self.xticks = list(ticks)

    def set_yticks(self, ticks: list[object]) -> None:
        self.yticks = list(ticks)

    def set_title(self, title: str) -> None:
        self.title = title

    def set_ylabel(self, ylabel: str) -> None:
        self.ylabel = ylabel

    def text(self, _x: float, _y: float, text: str, **_kwargs: object) -> None:
        self.text_calls.append(text)


class FakeFigure:
    def __init__(self, axes_grid: np.ndarray) -> None:
        self.axes_grid = axes_grid
        self.tight_layout_called = False

    def tight_layout(self) -> None:
        self.tight_layout_called = True


class FakePyplotModule(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")
        self.last_figure: FakeFigure | None = None

    def subplots(self, nrows: int, ncols: int, squeeze: bool = False) -> tuple[FakeFigure, np.ndarray]:
        assert not squeeze
        axes_grid = np.empty((nrows, ncols), dtype=object)
        for row_index in range(nrows):
            for column_index in range(ncols):
                axes_grid[row_index, column_index] = FakeAxis()
        figure = FakeFigure(axes_grid)
        self.last_figure = figure
        return figure, axes_grid
if __name__ == "__main__":
    unittest.main()
