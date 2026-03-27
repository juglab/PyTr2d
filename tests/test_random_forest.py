from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataio.projectio import build_frame_objects
from tracking.random_forest import (
    EventScorers,
    RandomForestEventTrainer,
    align_frame_to_gt,
    candidate_neighborhoods,
    load_event_scorers,
    save_event_scorers,
)


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


class RandomForestHelperTests(unittest.TestCase):
    def test_align_frame_to_gt_uses_one_to_one_iou_matching(self) -> None:
        raw = np.ones((6, 6), dtype=np.uint16)
        source_labels = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 2, 2],
                [0, 1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0],
                [0, 3, 3, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        gt_labels = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 11, 11, 0, 12, 12],
                [0, 11, 11, 0, 12, 12],
                [0, 0, 0, 0, 0, 0],
                [0, 13, 13, 0, 0, 0],
                [0, 13, 13, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        source_frame = build_frame_objects("source", 0, source_labels, raw)
        gt_frame = build_frame_objects("gt", 0, gt_labels, raw)

        matches = align_frame_to_gt(source_frame, gt_frame, iou_threshold=0.3)

        self.assertEqual(matches, {0: 11, 1: 12, 2: 13})

    def test_candidate_neighborhoods_respects_max_distance(self) -> None:
        raw = np.ones((8, 8), dtype=np.uint16)
        frame_a = build_frame_objects(
            "source",
            0,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            raw,
        )
        frame_b = build_frame_objects(
            "source",
            1,
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 3, 3, 0, 0, 0, 0],
                    [0, 0, 3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.uint16,
            ),
            raw,
        )

        neighborhoods = candidate_neighborhoods(frame_a, frame_b, max_distance=2.0)

        self.assertEqual(neighborhoods[0], [0])
        self.assertEqual(neighborhoods[1], [1])

    def test_save_then_load_event_scorers_round_trip(self) -> None:
        trainer = RandomForestEventTrainer(max_distance=12.0, train_iou_threshold=0.4)
        metadata = trainer.bundle_metadata("DemoDataset", "01")
        scorers = self._constant_scorers()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bundle.pkl"
            save_event_scorers(path, scorers, metadata)
            loaded = load_event_scorers(path, metadata)

        self.assertIsInstance(loaded, EventScorers)
        self.assertAlmostEqual(loaded.move_model.positive_probability, 0.9)
        self.assertAlmostEqual(loaded.division_model.positive_probability, 0.8)

    def test_load_event_scorers_rejects_incompatible_metadata(self) -> None:
        trainer = RandomForestEventTrainer(max_distance=12.0, train_iou_threshold=0.4)
        metadata = trainer.bundle_metadata("DemoDataset", "01")
        scorers = self._constant_scorers()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bundle.pkl"
            save_event_scorers(path, scorers, metadata)
            incompatible = dict(metadata)
            incompatible["train_sequence"] = "02"
            with self.assertRaises(ValueError):
                load_event_scorers(path, incompatible)

    def _constant_scorers(self) -> EventScorers:
        return EventScorers(
            move_model=ConstantProbabilityModel(0.9),
            division_model=ConstantProbabilityModel(0.8),
            appearance_model=ConstantProbabilityModel(0.7),
            disappearance_model=ConstantProbabilityModel(0.6),
        )


if __name__ == "__main__":
    unittest.main()
