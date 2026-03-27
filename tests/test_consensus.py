from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataio.projectio import build_frame_objects
from tracking.consensus import build_solution_index, solve_consensus_tracking
from tracking.random_forest import EventScorers
from tracking.types import SavedTrackingSolution, TrackingConfig


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


@unittest.skipUnless(_gurobi_available(), "Gurobi is required for consensus solver tests.")
class ConsensusSolverTests(unittest.TestCase):
    def test_consensus_writes_four_variants_and_metrics(self) -> None:
        raw_frames = np.stack([np.ones((6, 6), dtype=np.uint16) for _ in range(2)])
        embedseg_masks = np.stack(
            [
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
            ]
        )
        stardist_masks = np.stack(
            [
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
            ]
        )
        embedseg_solution = self._solution("embedseg", raw_frames, embedseg_masks)
        stardist_solution = self._solution("stardist", raw_frames, stardist_masks)
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.9),
            division_model=ConstantProbabilityModel(0.1),
            appearance_model=ConstantProbabilityModel(0.9),
            disappearance_model=ConstantProbabilityModel(0.9),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackingConfig(
                dataset_root=Path(tmpdir) / "dataset",
                extra_seg_root=None,
                mode="consensus",
                consensus_sources=("embedseg", "stardist"),
                consensus_output_dir=Path(tmpdir) / "consensus",
                log_file=Path(tmpdir) / "consensus" / "run.log",
            )
            config.dataset_root.mkdir(parents=True, exist_ok=True)

            result = solve_consensus_tracking(
                config,
                raw_frames,
                {"embedseg": embedseg_solution, "stardist": stardist_solution},
                scorers,
            )

            self.assertEqual(set(result.variant_evaluations), {"intersection", "union", "embedseg", "stardist"})
            for evaluation in result.variant_evaluations.values():
                self.assertTrue(evaluation.output_dir.exists())
                self.assertTrue(evaluation.metrics_json_path.exists())
                self.assertTrue(evaluation.metrics_text_path.exists())
                self.assertTrue(evaluation.lineage_path.exists())
                self.assertEqual(len(evaluation.mask_paths), 2)

            intersection_masks = result.variant_evaluations["intersection"].mask_paths
            union_masks = result.variant_evaluations["union"].mask_paths
            self.assertTrue(intersection_masks[0].exists())
            self.assertTrue(union_masks[0].exists())
            self.assertTrue(result.premerge_metrics_path.exists())
            self.assertTrue(result.variant_comparison_path.exists())

    def _solution(
        self,
        source_name: str,
        raw_frames: np.ndarray,
        tracked_masks: np.ndarray,
    ) -> SavedTrackingSolution:
        frames = tuple(
            build_frame_objects(source_name, frame_index, mask, raw_frames[frame_index])
            for frame_index, mask in enumerate(tracked_masks)
        )
        return SavedTrackingSolution(
            source_name=source_name,
            output_dir=Path("/tmp") / source_name,
            tracked_masks=tracked_masks,
            lineage_rows=(self._row(track_id=1, begin=0, end=1, parent=0),),
            frames=frames,
            checkpoint=None,
        )

    def _row(self, track_id: int, begin: int, end: int, parent: int):
        from tracking.types import LineageRecord

        return LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent)


if __name__ == "__main__":
    unittest.main()
