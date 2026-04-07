from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dataio import projectio
from dataio.projectio import build_frame_objects
from tracking.consensus import (
    ConsensusPreparation,
    ContinuationCandidate,
    ObjectStats,
    TrackletNode,
    build_common_overlap_candidate_pairs,
    build_render_manifest,
    build_solution_index,
    build_continuation_candidates,
    evaluate_saved_consensus_outputs,
    ctc_metric_deltas,
    filter_conflicting_hypotheses,
    candidate_oracle_metrics,
    geometry_option_names,
    geometry_source_agreement,
    gt_matches_for_nodes,
    render_variant_masks,
    summarize_fragment_graph,
    solve_consensus_tracking,
    write_render_manifest,
)
from tracking.random_forest import EventScorers
from tracking.types import CommonTracklet, ConsensusMetrics, LineageRecord, SavedTrackingSolution, TrackingConfig


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


class GeometryHelperTests(unittest.TestCase):
    def test_geometry_option_names_and_source_agreement(self) -> None:
        source_names = ("embedseg", "stardist")

        self.assertEqual(geometry_option_names(source_names), ("embedseg", "stardist", "intersection", "union"))
        self.assertAlmostEqual(geometry_source_agreement("embedseg", "embedseg", source_names), 1.0)
        self.assertAlmostEqual(geometry_source_agreement("embedseg", "stardist", source_names), 0.0)
        self.assertAlmostEqual(geometry_source_agreement("embedseg", "intersection", source_names), 0.5)
        self.assertAlmostEqual(geometry_source_agreement("intersection", "union", source_names), 1.0)

    def test_common_overlap_candidates_are_pruned_by_frame_overlap(self) -> None:
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=1,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(1, 1),
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
            TrackletNode(
                node_id=1,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=2,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
            TrackletNode(
                node_id=2,
                begin=3,
                end=3,
                fixed=False,
                kind="source_specific",
                source_name="stardist",
                source_track_id=3,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
            TrackletNode(
                node_id=3,
                begin=1,
                end=1,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(4, 4),
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
            TrackletNode(
                node_id=4,
                begin=3,
                end=3,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(5, 5),
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
        )

        common_source_candidates, common_common_candidates = build_common_overlap_candidate_pairs(
            nodes,
            frame_count=4,
            candidate_node_ids={0, 1, 2, 3, 4},
        )

        self.assertEqual(common_source_candidates[0], (1,))
        self.assertEqual(common_source_candidates[4], (2,))
        self.assertNotIn(3, common_source_candidates)
        self.assertEqual(common_common_candidates, ((0, 3),))


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
            self.assertIsNotNone(result.render_manifest_path)
            self.assertTrue(result.render_manifest_path.exists())

    def test_sparse_node_ids_do_not_break_variant_export(self) -> None:
        raw_frames = np.stack([np.ones((4, 4), dtype=np.uint16)])
        solution = self._solution(
            "embedseg",
            raw_frames,
            np.stack([np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)]),
        )
        scorers = EventScorers(
            move_model=ConstantProbabilityModel(0.9),
            division_model=ConstantProbabilityModel(0.1),
            appearance_model=ConstantProbabilityModel(0.9),
            disappearance_model=ConstantProbabilityModel(0.9),
        )
        preparation = ConsensusPreparation(
            common_tracklets=(CommonTracklet(tracklet_id=0, begin=0, end=0, source_names=("embedseg", "stardist"), source_track_ids=(1, 1)),),
            hypothesis_tracklets=(),
            nodes=(
                TrackletNode(
                    node_id=0,
                    begin=0,
                    end=0,
                    fixed=True,
                    kind="common",
                    source_name=None,
                    source_track_id=None,
                    source_names=("embedseg", "stardist"),
                    source_track_ids=(1, 1),
                    start_stats=ObjectStats(1.0, 1.0, 4.0, 0.0, 1.0),
                    end_stats=ObjectStats(1.0, 1.0, 4.0, 0.0, 1.0),
                ),
                TrackletNode(
                    node_id=5,
                    begin=0,
                    end=0,
                    fixed=False,
                    kind="hypothesis",
                    source_name="embedseg",
                    source_track_id=1,
                    source_names=None,
                    source_track_ids=None,
                    start_stats=ObjectStats(1.0, 1.0, 4.0, 0.0, 1.0),
                    end_stats=ObjectStats(1.0, 1.0, 4.0, 0.0, 1.0),
                ),
            ),
            matches_by_frame=(),
            input_metrics=ConsensusMetrics(
                agreement_metrics={},
                input_solution_metrics={},
                common_tracklet_count=1,
                hypothesis_tracklet_count=1,
            ),
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

            with mock.patch("tracking.consensus.prepare_consensus", return_value=preparation), \
                 mock.patch("tracking.consensus.solve_global_tracklet_ilp", return_value=({0, 5}, {0: None, 5: None}, {0: None, 5: None})), \
                 mock.patch("tracking.consensus.decode_selected_tracklets", return_value=((self._row(track_id=1, begin=0, end=0, parent=0),), {0: 1, 5: 1})), \
                 mock.patch("tracking.consensus.render_variant_masks", return_value=solution.tracked_masks), \
                 mock.patch("tracking.consensus.input_solution_metrics", return_value={}):
                result = solve_consensus_tracking(
                    config,
                    raw_frames,
                    {"embedseg": solution, "stardist": solution},
                    scorers,
                )

            self.assertIn("intersection", result.variant_evaluations)
            self.assertTrue(result.premerge_metrics_path.exists())

    def test_filter_conflicting_hypotheses_uses_fixed_common_union_overlap(self) -> None:
        raw_frames = np.stack([np.ones((5, 5), dtype=np.uint16)])
        embedseg_solution = self._solution(
            "embedseg",
            raw_frames,
            np.stack([np.array([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16)]),
        )
        stardist_solution = self._solution(
            "stardist",
            raw_frames,
            np.stack([np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16)]),
        )
        indexed_solutions = {
            "embedseg": build_solution_index(embedseg_solution),
            "stardist": build_solution_index(stardist_solution),
        }
        common_node = TrackletNode(
            node_id=0,
            begin=0,
            end=0,
            fixed=True,
            kind="common",
            source_name=None,
            source_track_id=None,
            source_names=("embedseg", "stardist"),
            source_track_ids=(1, 1),
            start_stats=ObjectStats(1.0, 1.5, 4.0, 0.0, 1.0),
            end_stats=ObjectStats(1.0, 1.5, 4.0, 0.0, 1.0),
        )
        overlapping_hypothesis = TrackletNode(
            node_id=1,
            begin=0,
            end=0,
            fixed=False,
            kind="hypothesis",
            source_name="stardist",
            source_track_id=1,
            source_names=None,
            source_track_ids=None,
            start_stats=ObjectStats(1.0, 2.5, 4.0, 0.0, 1.0),
            end_stats=ObjectStats(1.0, 2.5, 4.0, 0.0, 1.0),
        )
        non_overlapping_hypothesis = TrackletNode(
            node_id=2,
            begin=0,
            end=0,
            fixed=False,
            kind="hypothesis",
            source_name="stardist",
            source_track_id=2,
            source_names=None,
            source_track_ids=None,
            start_stats=ObjectStats(4.0, 4.0, 1.0, 0.0, 0.0),
            end_stats=ObjectStats(4.0, 4.0, 1.0, 0.0, 0.0),
        )
        # Inject a second non-overlapping stardist object for track 2.
        indexed_solutions["stardist"] = build_solution_index(
            SavedTrackingSolution(
                source_name="stardist",
                output_dir=Path("/tmp/stardist"),
                tracked_masks=np.stack([np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 2]], dtype=np.uint16)]),
                lineage_rows=(self._row(track_id=1, begin=0, end=0, parent=0), self._row(track_id=2, begin=0, end=0, parent=0)),
                frames=(build_frame_objects("stardist", 0, np.array([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 2]], dtype=np.uint16), raw_frames[0]),),
                checkpoint=None,
            )
        )

        kept = filter_conflicting_hypotheses(
            [common_node],
            [overlapping_hypothesis, non_overlapping_hypothesis],
            indexed_solutions,
        )

        self.assertEqual([node.node_id for node in kept], [2])

    def test_union_variant_clips_overlapping_fixed_common_masks(self) -> None:
        raw_frames = np.stack([np.ones((3, 4), dtype=np.uint16)])
        embedseg_solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 1, 2, 0],
                            [0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "embedseg",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0],
                            [0, 1, 2, 0],
                            [0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        stardist_solution = SavedTrackingSolution(
            source_name="stardist",
            output_dir=Path("/tmp/stardist"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "stardist",
                    0,
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        indexed_solutions = {
            "embedseg": build_solution_index(embedseg_solution),
            "stardist": build_solution_index(stardist_solution),
        }
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=True,
                kind="common",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(1, 1),
                start_stats=ObjectStats(1.0, 1.0, 1.0, 0.0, 1.0),
                end_stats=ObjectStats(1.0, 1.0, 1.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=1,
                begin=0,
                end=0,
                fixed=True,
                kind="common",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(2, 2),
                start_stats=ObjectStats(1.0, 2.0, 1.0, 0.0, 1.0),
                end_stats=ObjectStats(1.0, 2.0, 1.0, 0.0, 1.0),
            ),
        )

        tracked_masks = render_variant_masks(
            "union",
            raw_frames,
            nodes,
            {0, 1},
            {0: 1, 1: 2},
            indexed_solutions,
        )

        self.assertEqual(int(tracked_masks[0, 1, 1]), 1)
        self.assertEqual(int(tracked_masks[0, 1, 2]), 2)
        self.assertEqual(sorted(np.unique(tracked_masks[0][tracked_masks[0] > 0]).tolist()), [1, 2])

    def test_small_boundary_overlap_yields_tolerated_handoff_candidate(self) -> None:
        raw_frames = np.stack([np.ones((8, 12), dtype=np.uint16) for _ in range(2)])
        embedseg_masks = np.stack(
            [
                np.array(
                    [
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
                np.array(
                    [
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
            ]
        )
        stardist_masks = np.stack(
            [
                np.array(
                    [
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.uint16,
                ),
            ]
        )
        embedseg_solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=embedseg_masks,
            lineage_rows=(self._row(track_id=1, begin=0, end=1, parent=0),),
            frames=tuple(
                build_frame_objects("embedseg", frame_index, mask, raw_frames[frame_index])
                for frame_index, mask in enumerate(embedseg_masks)
            ),
            checkpoint=None,
        )
        stardist_solution = SavedTrackingSolution(
            source_name="stardist",
            output_dir=Path("/tmp/stardist"),
            tracked_masks=stardist_masks,
            lineage_rows=(self._row(track_id=1, begin=0, end=1, parent=0),),
            frames=tuple(
                build_frame_objects("stardist", frame_index, mask, raw_frames[frame_index])
                for frame_index, mask in enumerate(stardist_masks)
            ),
            checkpoint=None,
        )
        indexed = {
            "embedseg": build_solution_index(embedseg_solution),
            "stardist": build_solution_index(stardist_solution),
        }
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=1,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(2.0, 3.0, 25.0, 0.0, 1.0),
                end_stats=ObjectStats(2.0, 3.0, 25.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=1,
                begin=0,
                end=1,
                fixed=False,
                kind="source_specific",
                source_name="stardist",
                source_track_id=1,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(2.0, 7.2, 21.0, 0.0, 1.0),
                end_stats=ObjectStats(2.0, 7.2, 21.0, 0.0, 1.0),
            ),
        )
        config = TrackingConfig(dataset_root=Path("/tmp/dataset"), extra_seg_root=None, max_distance=10.0)

        candidates = build_continuation_candidates(config, nodes, indexed)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].is_tolerated_handoff)
        self.assertEqual(candidates[0].shared_boundary_frame, 0)

    def test_summary_graph_counts_tolerated_and_conflicts(self) -> None:
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(1, 1),
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                mean_iou=0.9,
                agreement_strength=0.9,
            ),
            TrackletNode(
                node_id=1,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=1,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
                end_stats=ObjectStats(0.0, 0.0, 1.0, 0.0, 0.0),
            ),
        )
        stats = summarize_fragment_graph(
            nodes=nodes,
            continuation_candidates=(
                ContinuationCandidate(parent_id=0, child_id=1, shared_boundary_frame=0, overlap_pixels=1, overlap_fraction=0.05),
            ),
            division_candidates=((0, 1, 1),),
            hard_conflicts=(frozenset((0, 1)),),
        )

        self.assertEqual(stats["common_supported_fragment_count"], 1)
        self.assertEqual(stats["source_specific_fragment_count"], 1)
        self.assertEqual(stats["tolerated_handoff_pair_count"], 1)
        self.assertEqual(stats["hard_conflict_count"], 1)

    def test_ctc_metric_deltas_are_computed_against_best_input(self) -> None:
        input_metrics = {
            "embedseg": {"ctc_evaluation": {"metrics": {"TRA": 0.95, "DET": 0.9}}},
            "stardist": {"ctc_evaluation": {"metrics": {"TRA": 0.92, "DET": 0.88}}},
        }

        deltas = ctc_metric_deltas(
            variant_ctc_payload={"metrics": {"TRA": 0.97, "DET": 0.91}},
            best_input_tra=0.95,
            best_input_source="embedseg",
            input_solution_metrics=input_metrics,
        )

        self.assertAlmostEqual(deltas["TRA"], 0.02)
        self.assertAlmostEqual(deltas["DET"], 0.01)
        self.assertAlmostEqual(deltas["delta_to_best_TRA"], 0.02)

    def test_gt_matches_for_nodes_uses_only_overlapping_gt_labels(self) -> None:
        raw_frames = np.stack([np.ones((5, 5), dtype=np.uint16)])
        solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(self._row(track_id=1, begin=0, end=0, parent=0),),
            frames=(
                build_frame_objects(
                    "embedseg",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        gt_solution = SavedTrackingSolution(
            source_name="gt",
            output_dir=Path("/tmp/gt"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 2, 0],
                            [0, 1, 1, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "gt",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 2, 0],
                            [0, 1, 1, 2, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        indexed = {"embedseg": build_solution_index(solution)}
        gt_indexed = build_solution_index(gt_solution)
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=1,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
            ),
        )

        matches = gt_matches_for_nodes(nodes, indexed, gt_indexed, 0.5)

        self.assertIn(1, matches[0])
        self.assertEqual(matches[0][1], {0})
        self.assertNotIn(2, matches[0])

    def test_two_stage_mode_writes_optimized_variant_and_assignments(self) -> None:
        raw_frames = np.stack([np.ones((4, 8), dtype=np.uint16) for _ in range(2)])
        embedseg_solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 3, 3, 0, 0, 0],
                            [0, 1, 1, 3, 3, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 0, 0, 0, 0, 0],
                            [0, 2, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=1, end=1, parent=0),
                self._row(track_id=3, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "embedseg",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 3, 3, 0, 0, 0],
                            [0, 1, 1, 3, 3, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
                build_frame_objects(
                    "embedseg",
                    1,
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 2, 0, 0, 0, 0, 0],
                            [0, 2, 2, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[1],
                ),
            ),
            checkpoint=None,
        )
        stardist_solution = SavedTrackingSolution(
            source_name="stardist",
            output_dir=Path("/tmp/stardist"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    np.zeros((4, 8), dtype=np.uint16),
                ]
            ),
            lineage_rows=(self._row(track_id=1, begin=0, end=0, parent=0),),
            frames=(
                build_frame_objects(
                    "stardist",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
                build_frame_objects("stardist", 1, np.zeros((4, 8), dtype=np.uint16), raw_frames[1]),
            ),
            checkpoint=None,
        )
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(1, 1),
                start_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=1,
                begin=1,
                end=1,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=2,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 1.5, 4.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=2,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=3,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(1.5, 3.5, 4.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 3.5, 4.0, 0.0, 1.0),
            ),
        )
        preparation = ConsensusPreparation(
            common_tracklets=(CommonTracklet(tracklet_id=0, begin=0, end=0, source_names=("embedseg", "stardist"), source_track_ids=(1, 1)),),
            hypothesis_tracklets=(),
            nodes=nodes,
            matches_by_frame=(),
            input_metrics=ConsensusMetrics(
                agreement_metrics={},
                input_solution_metrics={},
                common_tracklet_count=1,
                hypothesis_tracklet_count=2,
            ),
            graph_stats={},
            oracle_metrics={},
        )
        scorers = self._scorers(0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackingConfig(
                dataset_root=Path(tmpdir) / "dataset",
                extra_seg_root=None,
                mode="consensus",
                consensus_sources=("embedseg", "stardist"),
                consensus_output_dir=Path(tmpdir) / "consensus",
                log_file=Path(tmpdir) / "consensus" / "run.log",
                common_geometry_mode="two_stage",
            )
            config.dataset_root.mkdir(parents=True, exist_ok=True)

            with mock.patch("tracking.consensus.prepare_consensus", return_value=preparation), \
                 mock.patch(
                     "tracking.consensus.solve_global_tracklet_ilp",
                     return_value=(
                         {0, 1, 2},
                         {0: None, 1: ("move", 0), 2: None},
                         {0: ("move", 1), 1: None, 2: None},
                         {"selected_total_count": 3},
                     ),
                 ):
                result = solve_consensus_tracking(
                    config,
                    raw_frames,
                    {"embedseg": embedseg_solution, "stardist": stardist_solution},
                    scorers,
                )

            self.assertEqual(set(result.variant_evaluations), {"optimized_two_stage"})
            self.assertIsNotNone(result.geometry_assignments_path)
            payload = json.loads(result.geometry_assignments_path.read_text())
            self.assertEqual(payload["assignments"]["0"], "embedseg")
            self.assertTrue(result.render_manifest_path.exists())
            manifest = json.loads(result.render_manifest_path.read_text())
            self.assertEqual(manifest["variant_names"], ["optimized_two_stage"])

    def test_joint_mode_can_choose_intersection_to_keep_common_and_source_specific_fragments(self) -> None:
        raw_frames = np.stack([np.ones((4, 6), dtype=np.uint16)])
        embedseg_solution = SavedTrackingSolution(
            source_name="embedseg",
            output_dir=Path("/tmp/embedseg"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 2, 0, 0],
                            [0, 1, 1, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "embedseg",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 2, 0, 0],
                            [0, 1, 1, 2, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        stardist_solution = SavedTrackingSolution(
            source_name="stardist",
            output_dir=Path("/tmp/stardist"),
            tracked_masks=np.stack(
                [
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 2, 1, 1, 0, 0],
                            [0, 2, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    )
                ]
            ),
            lineage_rows=(
                self._row(track_id=1, begin=0, end=0, parent=0),
                self._row(track_id=2, begin=0, end=0, parent=0),
            ),
            frames=(
                build_frame_objects(
                    "stardist",
                    0,
                    np.array(
                        [
                            [0, 0, 0, 0, 0, 0],
                            [0, 2, 1, 1, 0, 0],
                            [0, 2, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                        ],
                        dtype=np.uint16,
                    ),
                    raw_frames[0],
                ),
            ),
            checkpoint=None,
        )
        nodes = (
            TrackletNode(
                node_id=0,
                begin=0,
                end=0,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=("embedseg", "stardist"),
                source_track_ids=(1, 1),
                start_stats=ObjectStats(1.5, 2.0, 4.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 2.0, 4.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=1,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="embedseg",
                source_track_id=2,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(1.5, 3.0, 2.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 3.0, 2.0, 0.0, 1.0),
            ),
            TrackletNode(
                node_id=2,
                begin=0,
                end=0,
                fixed=False,
                kind="source_specific",
                source_name="stardist",
                source_track_id=2,
                source_names=None,
                source_track_ids=None,
                start_stats=ObjectStats(1.5, 1.0, 2.0, 0.0, 1.0),
                end_stats=ObjectStats(1.5, 1.0, 2.0, 0.0, 1.0),
            ),
        )
        preparation = ConsensusPreparation(
            common_tracklets=(CommonTracklet(tracklet_id=0, begin=0, end=0, source_names=("embedseg", "stardist"), source_track_ids=(1, 1)),),
            hypothesis_tracklets=(),
            nodes=nodes,
            matches_by_frame=(),
            input_metrics=ConsensusMetrics(
                agreement_metrics={},
                input_solution_metrics={},
                common_tracklet_count=1,
                hypothesis_tracklet_count=2,
            ),
            graph_stats={},
            oracle_metrics={},
        )
        scorers = self._scorers(0.9)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrackingConfig(
                dataset_root=Path(tmpdir) / "dataset",
                extra_seg_root=None,
                mode="consensus",
                consensus_sources=("embedseg", "stardist"),
                consensus_output_dir=Path(tmpdir) / "consensus",
                log_file=Path(tmpdir) / "consensus" / "run.log",
                common_geometry_mode="joint",
            )
            config.dataset_root.mkdir(parents=True, exist_ok=True)

            with mock.patch("tracking.consensus.prepare_consensus", return_value=preparation):
                result = solve_consensus_tracking(
                    config,
                    raw_frames,
                    {"embedseg": embedseg_solution, "stardist": stardist_solution},
                    scorers,
                )

            self.assertEqual(set(result.variant_evaluations), {"optimized_joint"})
            self.assertIsNotNone(result.geometry_assignments_path)
            payload = json.loads(result.geometry_assignments_path.read_text())
            self.assertEqual(payload["assignments"]["0"], "intersection")

    def test_evaluate_only_uses_render_manifest_variant_names(self) -> None:
        raw_frames = np.stack([np.ones((3, 3), dtype=np.uint16)])
        embedseg_solution = self._solution(
            "embedseg",
            raw_frames,
            np.stack([np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint16)]),
        )
        stardist_solution = self._solution(
            "stardist",
            raw_frames,
            np.stack([np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint16)]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "consensus"
            config = TrackingConfig(
                dataset_root=Path(tmpdir) / "dataset",
                extra_seg_root=None,
                mode="consensus",
                consensus_sources=("embedseg", "stardist"),
                consensus_output_dir=output_root,
                common_geometry_mode="two_stage",
            )
            config.dataset_root.mkdir(parents=True, exist_ok=True)
            variant_name = "optimized_two_stage"
            projectio.write_tracking_outputs(
                output_root / variant_name,
                embedseg_solution.tracked_masks,
                (LineageRecord(track_id=1, begin=0, end=0, parent=0),),
            )
            (output_root / variant_name / "metrics.json").write_text(
                json.dumps({"optimized_geometry": {"assignment_count_by_option": {"embedseg": 1}}}),
                encoding="utf-8",
            )
            (output_root / "geometry_assignments.json").write_text("{}", encoding="utf-8")
            write_render_manifest(
                output_root,
                build_render_manifest(
                    config=config,
                    variant_names=(variant_name,),
                    geometry_assignments_filename="geometry_assignments.json",
                ),
            )

            result = evaluate_saved_consensus_outputs(
                config,
                raw_frames,
                {"embedseg": embedseg_solution, "stardist": stardist_solution},
            )

            self.assertEqual(set(result.variant_evaluations), {variant_name})
            self.assertEqual(result.geometry_assignments_path, output_root / "geometry_assignments.json")
            self.assertTrue(result.render_manifest_path.exists())

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
            lineage_rows=(self._row(track_id=1, begin=0, end=len(tracked_masks) - 1, parent=0),),
            frames=frames,
            checkpoint=None,
        )

    def _row(self, track_id: int, begin: int, end: int, parent: int):
        return LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent)

    def _scorers(self, positive_probability: float) -> EventScorers:
        return EventScorers(
            move_model=ConstantProbabilityModel(positive_probability),
            division_model=ConstantProbabilityModel(positive_probability),
            appearance_model=ConstantProbabilityModel(positive_probability),
            disappearance_model=ConstantProbabilityModel(positive_probability),
        )


if __name__ == "__main__":
    unittest.main()
