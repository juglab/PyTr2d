from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from main import _run_consensus_tracking, load_or_train_event_scorers, run_tracking
from tracking.random_forest import EventScorers, RandomForestEventTrainer, save_event_scorers
from tracking.types import ConsensusResult, SavedTrackingSolution, TrackingConfig, TrackingResult, VariantEvaluation


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


class ModelPersistenceFlowTests(unittest.TestCase):
    def test_force_retrain_bypasses_saved_bundle(self) -> None:
        trainer = RandomForestEventTrainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackingConfig(
                dataset_root=Path(tmpdir) / "dataset",
                extra_seg_root=None,
                model_dir=Path(tmpdir) / "models",
                force_retrain=True,
            )
            config.dataset_root.mkdir()
            saved = self._scorers(0.2)
            replacement = self._scorers(0.9)
            save_event_scorers(
                config.model_dir / config.dataset_root.name / config.train_sequence / "event_scorers.pkl",
                saved,
                trainer.bundle_metadata(config.dataset_root.name, config.train_sequence),
            )

            fit_calls = {"count": 0}

            def fit_scorers() -> EventScorers:
                fit_calls["count"] += 1
                return replacement

            loaded = load_or_train_event_scorers(config, trainer, fit_scorers)

        self.assertEqual(fit_calls["count"], 1)
        self.assertAlmostEqual(loaded.move_model.positive_probability, 0.9)

    def _scorers(self, positive_probability: float) -> EventScorers:
        return EventScorers(
            move_model=ConstantProbabilityModel(positive_probability),
            division_model=ConstantProbabilityModel(positive_probability),
            appearance_model=ConstantProbabilityModel(positive_probability),
            disappearance_model=ConstantProbabilityModel(positive_probability),
        )


class ConsensusMainFlowTests(unittest.TestCase):
    def test_consensus_runs_only_missing_source_results(self) -> None:
        raw_frames = np.stack([np.ones((4, 4), dtype=np.uint16) for _ in range(2)])
        fake_solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=np.zeros((2, 4, 4), dtype=np.uint16),
            lineage_rows=(),
            frames=(),
            checkpoint=None,
        )
        fake_result = ConsensusResult(
            selected_sources=("embedseg", "stardist"),
            lineage_rows=(),
            output_dir=Path("/tmp/consensus"),
            premerge_metrics_path=Path("/tmp/consensus/premerge_metrics.json"),
            premerge_metrics_text_path=Path("/tmp/consensus/premerge_metrics.txt"),
            variant_comparison_path=Path("/tmp/consensus/variant_comparison.json"),
            variant_comparison_text_path=Path("/tmp/consensus/variant_comparison.txt"),
            variant_evaluations={
                "optimized_joint": VariantEvaluation(
                    variant_name="optimized_joint",
                    metrics={},
                    output_dir=Path("/tmp/consensus/optimized_joint"),
                    mask_paths=(),
                    lineage_path=Path("/tmp/consensus/optimized_joint/res_track.txt"),
                    metrics_json_path=Path("/tmp/consensus/optimized_joint/metrics.json"),
                    metrics_text_path=Path("/tmp/consensus/optimized_joint/metrics.txt"),
                )
            },
        )
        config = TrackingConfig(
            dataset_root=Path("/tmp/dataset"),
            extra_seg_root=None,
            mode="consensus",
            consensus_sources=("embedseg", "stardist"),
            log_file=Path("/tmp/consensus/run.log"),
        )
        scorers = self._scorers(0.9)

        with mock.patch("main.projectio.load_raw_sequence", return_value=raw_frames), \
             mock.patch("main.projectio.resolve_source_output_dir", side_effect=[Path("/tmp/embedseg"), Path("/tmp/stardist")]), \
             mock.patch("main.projectio.is_saved_tracking_complete", side_effect=[True, False]), \
             mock.patch("main.projectio.load_saved_tracking_solution", side_effect=[fake_solution, fake_solution]), \
             mock.patch("main._run_single_tracking") as single_run, \
             mock.patch("main.solve_consensus_tracking", return_value=fake_result):
            result = _run_consensus_tracking(config, scorers)

        self.assertIs(result, fake_result)
        self.assertEqual(single_run.call_count, 1)
        single_config = single_run.call_args.args[0]
        self.assertEqual(single_config.seg_source, "stardist")

    def _scorers(self, positive_probability: float) -> EventScorers:
        return EventScorers(
            move_model=ConstantProbabilityModel(positive_probability),
            division_model=ConstantProbabilityModel(positive_probability),
            appearance_model=ConstantProbabilityModel(positive_probability),
            disappearance_model=ConstantProbabilityModel(positive_probability),
        )


class EvaluateOnlyRoutingTests(unittest.TestCase):
    def test_single_evaluate_only_bypasses_scorer_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()
            config = TrackingConfig(
                dataset_root=dataset_root,
                extra_seg_root=None,
                mode="single",
                evaluate_only=True,
                seg_source="stardist",
            )
            fake_result = TrackingResult(
                selected_sources=("stardist",),
                tracklets=(),
                lineage_rows=(),
                tracked_masks=np.zeros((1, 2, 2), dtype=np.uint16),
            )

            with mock.patch("main._evaluate_single_tracking", return_value=fake_result) as evaluate_single, \
                 mock.patch("main.load_or_train_event_scorers") as load_scorers:
                result = run_tracking(config)

        self.assertIs(result, fake_result)
        evaluate_single.assert_called_once_with(config)
        load_scorers.assert_not_called()

    def test_consensus_evaluate_only_bypasses_scorer_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()
            config = TrackingConfig(
                dataset_root=dataset_root,
                extra_seg_root=None,
                mode="consensus",
                evaluate_only=True,
                consensus_sources=("embedseg", "stardist"),
            )
            fake_result = ConsensusResult(
                selected_sources=("embedseg", "stardist"),
                lineage_rows=(),
                output_dir=Path("/tmp/consensus"),
                premerge_metrics_path=Path("/tmp/consensus/premerge_metrics.json"),
                premerge_metrics_text_path=Path("/tmp/consensus/premerge_metrics.txt"),
                variant_comparison_path=Path("/tmp/consensus/variant_comparison.json"),
                variant_comparison_text_path=Path("/tmp/consensus/variant_comparison.txt"),
                variant_evaluations={},
            )

            with mock.patch("main._evaluate_consensus_tracking", return_value=fake_result) as evaluate_consensus, \
                 mock.patch("main.load_or_train_event_scorers") as load_scorers:
                result = run_tracking(config)

        self.assertIs(result, fake_result)
        evaluate_consensus.assert_called_once_with(config)
        load_scorers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
