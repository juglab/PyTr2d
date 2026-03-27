from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from dataio import projectio
from dataio.projectio import build_frame_objects
from tracking.random_forest import EventScorers
from tracking.trackingsolver import solve_tracking
from tracking.types import FrameObjects, LineageRecord, TrackingCheckpoint, TrackingConfig


def _gurobi_available() -> bool:
    try:
        import gurobipy as gp

        model = gp.Model()
        model.Params.OutputFlag = 0
        variable = model.addVar(vtype=gp.GRB.BINARY)
        model.setObjective(variable, gp.GRB.MAXIMIZE)
        model.optimize()
        return model.Status in {gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL, gp.GRB.TIME_LIMIT}
    except Exception:
        return False


class ConstantProbabilityModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability
        self.classes_ = np.array([0, 1])

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs)
        probabilities = np.empty((len(inputs), 2), dtype=float)
        probabilities[:, 0] = 1.0 - self.positive_probability
        probabilities[:, 1] = self.positive_probability
        return probabilities


@unittest.skipUnless(_gurobi_available(), "Gurobi is required for tracking solver tests.")
class TrackingSolverTests(unittest.TestCase):
    def test_solver_carries_track_id_across_three_frames_when_moves_are_likely(self) -> None:
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.99),
            division_model=ConstantProbabilityModel(0.01),
            appearance_model=ConstantProbabilityModel(0.01),
            disappearance_model=ConstantProbabilityModel(0.01),
        )
        raw_frames = self._raw_stack(3, (5, 5))
        frames_by_source = {
            "st": [
                self._frame("st", np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16), 0),
                self._frame("st", np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16), 1),
                self._frame("st", np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16), 2),
            ]
        }

        result = solve_tracking(TrackingConfig(dataset_root=Path("."), extra_seg_root=None), raw_frames, frames_by_source, scorers)

        self.assertEqual(len(result.lineage_rows), 1)
        self.assertEqual(result.lineage_rows[0].begin, 0)
        self.assertEqual(result.lineage_rows[0].end, 2)
        self.assertEqual(np.max(result.tracked_masks[0]), 1)
        self.assertEqual(np.max(result.tracked_masks[2]), 1)

    def test_solver_creates_children_on_division(self) -> None:
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.2),
            division_model=ConstantProbabilityModel(0.99),
            appearance_model=ConstantProbabilityModel(0.01),
            disappearance_model=ConstantProbabilityModel(0.01),
        )
        raw_frames = self._raw_stack(2, (5, 6))
        frames_by_source = {
            "st": [
                self._frame(
                    "st",
                    np.array(
                        [
                            [0, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    0,
                ),
                self._frame(
                    "st",
                    np.array(
                        [
                            [0, 2, 2, 0, 3, 3],
                            [0, 2, 2, 0, 3, 3],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    1,
                ),
            ]
        }

        result = solve_tracking(
            TrackingConfig(dataset_root=Path("."), extra_seg_root=None, max_distance=6.0),
            raw_frames,
            frames_by_source,
            scorers,
        )

        self.assertEqual(len(result.lineage_rows), 3)
        parent_row = result.lineage_rows[0]
        child_rows = result.lineage_rows[1:]
        self.assertEqual(parent_row.track_id, 1)
        self.assertEqual(parent_row.begin, 0)
        self.assertEqual(parent_row.end, 0)
        self.assertEqual(sorted(row.parent for row in child_rows), [1, 1])

    def test_overlap_constraint_keeps_only_one_overlapping_source_at_initialization(self) -> None:
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.5),
            division_model=ConstantProbabilityModel(0.5),
            appearance_model=ConstantProbabilityModel(0.5),
            disappearance_model=ConstantProbabilityModel(0.5),
        )
        overlapping = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint16)
        frames_by_source = {
            "st": [self._frame("st", overlapping, 0)],
            "err_seg": [self._frame("err_seg", overlapping, 0)],
        }
        raw_frames = self._raw_stack(1, overlapping.shape)

        result = solve_tracking(TrackingConfig(dataset_root=Path("."), extra_seg_root=None), raw_frames, frames_by_source, scorers)

        self.assertEqual(len(result.lineage_rows), 1)
        self.assertEqual(np.max(result.tracked_masks), 1)

    def test_solver_handles_appearance_and_disappearance(self) -> None:
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.01),
            division_model=ConstantProbabilityModel(0.01),
            appearance_model=ConstantProbabilityModel(0.99),
            disappearance_model=ConstantProbabilityModel(0.99),
        )
        raw_frames = self._raw_stack(2, (4, 4))
        frames_by_source = {
            "st": [
                self._frame("st", np.zeros((4, 4), dtype=np.uint16), 0),
                self._frame("st", np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint16), 1),
            ]
        }

        result = solve_tracking(TrackingConfig(dataset_root=Path("."), extra_seg_root=None), raw_frames, frames_by_source, scorers)

        self.assertEqual(len(result.lineage_rows), 1)
        self.assertEqual(result.lineage_rows[0].begin, 1)
        self.assertEqual(result.lineage_rows[0].end, 1)
        self.assertEqual(result.lineage_rows[0].parent, 0)

    def test_solver_resumes_from_saved_checkpoint(self) -> None:
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.99),
            division_model=ConstantProbabilityModel(0.01),
            appearance_model=ConstantProbabilityModel(0.01),
            disappearance_model=ConstantProbabilityModel(0.01),
        )
        raw_frames = self._raw_stack(3, (5, 5))
        frame0 = np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16)
        frames_by_source = {
            "st": [
                self._frame("st", frame0, 0),
                self._frame("st", np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16), 1),
                self._frame("st", np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16), 2),
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = TrackingConfig(dataset_root=Path("."), extra_seg_root=None, output_dir=output_dir)
            projectio.write_tracking_mask(output_dir, 0, frame0)
            projectio.write_lineage_rows(output_dir, (LineageRecord(track_id=1, begin=0, end=0, parent=0),))
            projectio.write_tracking_checkpoint(
                output_dir,
                TrackingCheckpoint(
                    version=1,
                    dataset_root=str(config.dataset_root),
                    track_sequence=config.track_sequence,
                    seg_source=config.seg_source,
                    selected_sources=("st",),
                    frame_count=3,
                    frame_shape=(5, 5),
                    completed_frame=0,
                    next_track_id=2,
                    max_distance=config.max_distance,
                    segmentation_reward=config.segmentation_reward,
                    lineage_state={1: (0, 0, 0)},
                ),
            )

            with mock.patch("tracking.trackingsolver.initialize_first_frame", side_effect=AssertionError("should resume")):
                result = solve_tracking(config, raw_frames, frames_by_source, scorers)

        self.assertEqual(len(result.lineage_rows), 1)
        self.assertEqual(result.lineage_rows[0].end, 2)

    def _frame(self, source_name: str, labels: np.ndarray, frame_index: int) -> FrameObjects:
        raw = np.ones_like(labels, dtype=np.uint16)
        return build_frame_objects(source_name, frame_index, labels, raw)

    def _raw_stack(self, frame_count: int, shape: tuple[int, int]) -> np.ndarray:
        return np.stack([np.ones(shape, dtype=np.uint16) for _ in range(frame_count)])


if __name__ == "__main__":
    unittest.main()
