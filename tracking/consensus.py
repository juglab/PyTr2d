from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised if tqdm is not installed.
    tqdm = None

try:
    import gurobipy as gp
    from gurobipy import GRB, quicksum
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without Gurobi.
    gp = None
    GRB = None
    quicksum = None

from dataio import projectio
from tracking import reporting
from tracking.ctc_evaluation import evaluate_result_with_ctc, log_ctc_evaluation
from tracking.random_forest import EventScorers, intersection_areas, probability_to_cost
from tracking.types import (
    CommonTracklet,
    ConsensusMetrics,
    ConsensusResult,
    HypothesisTracklet,
    LineageRecord,
    MatchedObjectPair,
    SavedTrackingSolution,
    TrackingConfig,
    VariantEvaluation,
)


LOGGER = logging.getLogger(__name__)
GT_EVAL_IOU_THRESHOLD = 0.5
ORACLE_IOU_THRESHOLD = 0.5
COMMON_FRAGMENT_BONUS_SCALE = 18.0
SOURCE_FRAGMENT_BONUS_SCALE = 10.0
BOUNDARY_PENALTY_SCALE = 1.25
HANDOFF_OVERLAP_ABSOLUTE_PIXELS = 5
HANDOFF_OVERLAP_RELATIVE_FRACTION = 0.05
PROGRESS_LOG_STEPS = 20
PROGRESS_LOG_FALLBACK_EVERY = 250
RENDER_MANIFEST_FILENAME = "render_manifest.json"
GEOMETRY_ASSIGNMENTS_FILENAME = "geometry_assignments.json"
POSTHOC_GEOMETRY_MODE = "posthoc"
TWO_STAGE_GEOMETRY_MODE = "two_stage"
JOINT_GEOMETRY_MODE = "joint"
OPTIMIZED_TWO_STAGE_VARIANT = "optimized_two_stage"
OPTIMIZED_JOINT_VARIANT = "optimized_joint"


@dataclass(slots=True)
class ProgressTracker:
    desc: str
    total: int | None = None
    unit: str = "item"
    log_every: int | None = None
    count: int = field(init=False, default=0)
    next_log: int = field(init=False, default=0)
    started_at: float = field(init=False, default_factory=time.monotonic)
    progress_bar: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.log_every is None:
            if self.total is None:
                self.log_every = PROGRESS_LOG_FALLBACK_EVERY
            else:
                self.log_every = max(1, self.total // PROGRESS_LOG_STEPS)
        self.next_log = self.log_every
        total_suffix = f" ({self.total} {self.unit}{'' if self.total == 1 else 's'})" if self.total is not None else ""
        LOGGER.info("%s started%s.", self.desc, total_suffix)
        if tqdm is not None and sys.stderr.isatty():
            self.progress_bar = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                leave=False,
                dynamic_ncols=True,
            )

    def advance(self, step: int = 1) -> None:
        self.count += step
        if self.progress_bar is not None:
            self.progress_bar.update(step)
        if self.log_every is None:
            return
        if self.total is None:
            if self.count >= self.next_log:
                LOGGER.info("%s progress: %s %s%s processed.", self.desc, self.count, self.unit, "" if self.count == 1 else "s")
                self.next_log += self.log_every
            return
        while self.count >= self.next_log and self.next_log < self.total:
            LOGGER.info(
                "%s progress: %s/%s %s%s (%.1f%%).",
                self.desc,
                self.count,
                self.total,
                self.unit,
                "" if self.total == 1 else "s",
                100.0 * self.count / max(1, self.total),
            )
            self.next_log += self.log_every

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
        elapsed = time.monotonic() - self.started_at
        if self.total is None:
            LOGGER.info("%s complete: %s %s%s processed in %.1fs.", self.desc, self.count, self.unit, "" if self.count == 1 else "s", elapsed)
            return
        LOGGER.info(
            "%s complete: %s/%s %s%s in %.1fs.",
            self.desc,
            self.count,
            self.total,
            self.unit,
            "" if self.total == 1 else "s",
            elapsed,
        )


def progress_iter(
    iterable,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    log_every: int | None = None,
):
    tracker = ProgressTracker(desc=desc, total=total, unit=unit, log_every=log_every)
    try:
        for item in iterable:
            yield item
            tracker.advance()
    finally:
        tracker.close()


def log_consensus_stage(stage_number: int, stage_total: int, message: str) -> None:
    LOGGER.info("Consensus prep %s/%s: %s", stage_number, stage_total, message)


def _assert_usable_status(model: gp.Model) -> None:
    if GRB is None:
        raise RuntimeError("Gurobi is unavailable.")
    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi failed to find a usable solution. Status code: {model.Status}")


@dataclass(slots=True, frozen=True)
class ObjectStats:
    centroid_row: float
    centroid_col: float
    area: float
    intensity_std: float
    border_distance: float


@dataclass(slots=True, frozen=True)
class SolutionIndex:
    solution: SavedTrackingSolution
    rows_by_track: dict[int, LineageRecord]
    children_by_parent: dict[int, tuple[int, ...]]


@dataclass(slots=True, frozen=True)
class FrameMatches:
    pairs: tuple[MatchedObjectPair, ...]
    source_1_to_2: dict[int, int]
    source_2_to_1: dict[int, int]
    pair_iou: dict[tuple[int, int], float]


@dataclass(slots=True, frozen=True)
class TrackletNode:
    node_id: int
    begin: int
    end: int
    fixed: bool
    kind: str
    source_name: str | None
    source_track_id: int | None
    source_names: tuple[str, str] | None
    source_track_ids: tuple[int, int] | None
    start_stats: ObjectStats
    end_stats: ObjectStats
    mean_iou: float = 0.0
    agreement_strength: float = 0.0

    @property
    def frame_count(self) -> int:
        return self.end - self.begin + 1

    @property
    def is_common_supported(self) -> bool:
        return self.kind in {"common_supported", "common"} or self.fixed or self.source_names is not None


@dataclass(slots=True, frozen=True)
class ContinuationCandidate:
    parent_id: int
    child_id: int
    shared_boundary_frame: int | None
    overlap_pixels: int
    overlap_fraction: float

    @property
    def pair_key(self) -> frozenset[int]:
        return frozenset((self.parent_id, self.child_id))

    @property
    def is_tolerated_handoff(self) -> bool:
        return self.shared_boundary_frame is not None


@dataclass(slots=True, frozen=True)
class ConsensusPreparation:
    common_tracklets: tuple[CommonTracklet, ...]
    hypothesis_tracklets: tuple[HypothesisTracklet, ...]
    nodes: tuple[TrackletNode, ...]
    matches_by_frame: tuple[FrameMatches, ...]
    input_metrics: ConsensusMetrics
    continuation_candidates: tuple[ContinuationCandidate, ...] = ()
    division_candidates: tuple[tuple[int, int, int], ...] = ()
    hard_conflicts: tuple[frozenset[int], ...] = ()
    overlap_pairs: tuple[frozenset[int], ...] = ()
    graph_stats: dict[str, object] = field(default_factory=dict)
    oracle_metrics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class NodeFrameIndex:
    source_nodes_by_frame_track: dict[str, tuple[dict[int, int], ...]]
    common_nodes_by_frame_source_track: dict[str, tuple[dict[int, set[int]], ...]]
    common_nodes_by_frame: tuple[tuple[int, ...], ...]
    cross_overlap_by_frame: tuple[dict[tuple[int, int], int], ...]


@dataclass(slots=True, frozen=True)
class FragmentDiagnostics:
    best_input_source: str | None
    best_input_tra: float | None
    graph_statistics: dict[str, object]
    oracle_metrics: dict[str, object]
    variant_deltas_to_best_input: dict[str, dict[str, float]]


@dataclass(slots=True, frozen=True)
class GeometryOptimizationResult:
    variant_name: str
    common_geometry_assignments: dict[int, str]
    diagnostics: dict[str, object]


@dataclass(slots=True)
class GeometryQueryCache:
    coords_by_key: dict[tuple[int, str, int], np.ndarray] = field(default_factory=dict)
    bbox_by_key: dict[tuple[int, str, int], tuple[int, int, int, int] | None] = field(default_factory=dict)
    coord_set_by_key: dict[tuple[int, str, int], frozenset[tuple[int, int]]] = field(default_factory=dict)


def default_consensus_variant_names(source_names: tuple[str, str]) -> tuple[str, ...]:
    return ("intersection", "union", source_names[0], source_names[1])


def geometry_option_names(source_names: tuple[str, str]) -> tuple[str, ...]:
    return (source_names[0], source_names[1], "intersection", "union")


def optimized_variant_name(common_geometry_mode: str) -> str:
    if common_geometry_mode == TWO_STAGE_GEOMETRY_MODE:
        return OPTIMIZED_TWO_STAGE_VARIANT
    if common_geometry_mode == JOINT_GEOMETRY_MODE:
        return OPTIMIZED_JOINT_VARIANT
    raise ValueError(f"Optimized geometry mode expected, found '{common_geometry_mode}'.")


def render_manifest_path(output_root: Path) -> Path:
    return output_root / RENDER_MANIFEST_FILENAME


def geometry_assignments_path(output_root: Path) -> Path:
    return output_root / GEOMETRY_ASSIGNMENTS_FILENAME


def build_render_manifest(
    config: TrackingConfig,
    variant_names: tuple[str, ...],
    geometry_assignments_filename: str | None = None,
) -> dict[str, object]:
    return {
        "common_geometry_mode": config.common_geometry_mode,
        "variant_names": list(variant_names),
        "consensus_sources": list(config.consensus_sources),
        "geometry_source_weight": float(config.geometry_source_weight),
        "geometry_temporal_overlap_weight": float(config.geometry_temporal_overlap_weight),
        "geometry_neighbor_radius": int(config.geometry_neighbor_radius),
        "geometry_assignments_filename": geometry_assignments_filename,
    }


def write_render_manifest(
    output_root: Path,
    payload: dict[str, object],
) -> Path:
    path = render_manifest_path(output_root)
    reporting.write_json(path, payload)
    return path


def load_render_manifest(output_root: Path) -> dict[str, object] | None:
    path = render_manifest_path(output_root)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid render manifest structure in {path}.")
    return payload


def render_variant_names_from_manifest(
    manifest: dict[str, object] | None,
    source_names: tuple[str, str],
) -> tuple[str, ...]:
    if manifest is None:
        return default_consensus_variant_names(source_names)
    variant_names = manifest.get("variant_names")
    if isinstance(variant_names, list) and all(isinstance(value, str) for value in variant_names):
        return tuple(str(value) for value in variant_names)
    return default_consensus_variant_names(source_names)


def saved_geometry_assignments_path(
    output_root: Path,
    manifest: dict[str, object] | None,
) -> Path | None:
    if manifest is None:
        return None
    filename = manifest.get("geometry_assignments_filename")
    if isinstance(filename, str) and filename:
        return output_root / filename
    return None


def load_saved_variant_geometry_metrics(
    output_root: Path,
    variant_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for variant_name in variant_names:
        metrics_path = output_root / variant_name / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if isinstance(metrics, dict):
            optimized_geometry = metrics.get("optimized_geometry")
            if isinstance(optimized_geometry, dict):
                payloads[variant_name] = optimized_geometry
    return payloads


def geometry_support(option_name: str, source_names: tuple[str, str]) -> frozenset[str]:
    if option_name == "intersection" or option_name == "union":
        return frozenset(source_names)
    if option_name == source_names[0]:
        return frozenset((source_names[0],))
    if option_name == source_names[1]:
        return frozenset((source_names[1],))
    raise ValueError(f"Unknown geometry option '{option_name}'.")


def geometry_source_agreement(
    left_option: str,
    right_option: str,
    source_names: tuple[str, str],
) -> float:
    left_support = geometry_support(left_option, source_names)
    right_support = geometry_support(right_option, source_names)
    union = left_support | right_support
    if not union:
        return 1.0
    return float(len(left_support & right_support) / len(union))


def solve_consensus_tracking(
    config: TrackingConfig,
    raw_frames: np.ndarray,
    solutions_by_source: dict[str, SavedTrackingSolution],
    scorers: EventScorers,
) -> ConsensusResult:
    if len(solutions_by_source) != 2:
        raise ValueError("Consensus tracking currently expects exactly two saved source solutions.")

    source_names = tuple(config.consensus_sources)
    if set(source_names) != set(solutions_by_source):
        raise ValueError(
            f"Consensus sources {source_names} do not match loaded solutions {tuple(sorted(solutions_by_source))}."
        )

    LOGGER.info("Preparing consensus tracking from sources: %s.", ", ".join(source_names))
    indexed_solutions = {name: build_solution_index(solution) for name, solution in solutions_by_source.items()}
    preparation = prepare_consensus(config, indexed_solutions, scorers)
    output_root = projectio.resolve_consensus_output_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Consensus preparation produced %s common-supported fragment(s) and %s source-specific fragment(s).",
        len(preparation.common_tracklets),
        len(preparation.hypothesis_tracklets),
    )
    premerge_json_path = output_root / "premerge_metrics.json"
    premerge_text_path = output_root / "premerge_metrics.txt"
    premerge_payload = {
        "agreement_metrics": preparation.input_metrics.agreement_metrics,
        "input_solution_metrics": preparation.input_metrics.input_solution_metrics,
        "common_tracklet_count": preparation.input_metrics.common_tracklet_count,
        "hypothesis_tracklet_count": preparation.input_metrics.hypothesis_tracklet_count,
        "graph_statistics": preparation.graph_stats,
        "candidate_oracle": preparation.oracle_metrics,
    }
    reporting.write_json(premerge_json_path, premerge_payload)
    reporting.write_text(premerge_text_path, reporting.format_metrics_report(premerge_payload))
    LOGGER.info("Wrote pre-merge metrics to %s and %s.", premerge_json_path, premerge_text_path)

    geometry_result: GeometryOptimizationResult | None = None
    if config.common_geometry_mode == JOINT_GEOMETRY_MODE:
        (
            selected_nodes,
            incoming_choice,
            outgoing_choice,
            selected_graph_stats,
            common_geometry_assignments,
            geometry_diagnostics,
        ) = solve_joint_tracklet_geometry_ilp(
            config=config,
            preparation=preparation,
            indexed_solutions=indexed_solutions,
            scorers=scorers,
        )
        geometry_result = GeometryOptimizationResult(
            variant_name=optimized_variant_name(config.common_geometry_mode),
            common_geometry_assignments=common_geometry_assignments,
            diagnostics=geometry_diagnostics,
        )
    else:
        solver_result = solve_global_tracklet_ilp(
            config=config,
            preparation=preparation,
            indexed_solutions=indexed_solutions,
            scorers=scorers,
        )
        if len(solver_result) == 3:
            selected_nodes, incoming_choice, outgoing_choice = solver_result
            selected_graph_stats = {}
        else:
            selected_nodes, incoming_choice, outgoing_choice, selected_graph_stats = solver_result
    lineage_rows, node_to_final_track = decode_selected_tracklets(
        nodes=preparation.nodes,
        selected_nodes=selected_nodes,
        incoming_choice=incoming_choice,
        outgoing_choice=outgoing_choice,
    )

    if config.common_geometry_mode == TWO_STAGE_GEOMETRY_MODE:
        geometry_result = solve_two_stage_geometry_ilp(
            config=config,
            preparation=preparation,
            indexed_solutions=indexed_solutions,
            selected_nodes=selected_nodes,
            incoming_choice=incoming_choice,
            outgoing_choice=outgoing_choice,
        )

    tracked_masks_by_variant: dict[str, np.ndarray]
    optimized_geometry_payloads: dict[str, dict[str, object]] = {}
    geometry_assignments_json_path: Path | None = None
    if geometry_result is None:
        tracked_masks_by_variant = {
            variant_name: render_variant_masks(
                variant_name=variant_name,
                raw_frames=raw_frames,
                nodes=preparation.nodes,
                selected_nodes=selected_nodes,
                node_to_final_track=node_to_final_track,
                indexed_solutions=indexed_solutions,
            )
            for variant_name in default_consensus_variant_names(source_names)
        }
    else:
        tracked_masks_by_variant = {
            geometry_result.variant_name: render_variant_masks(
                variant_name=geometry_result.variant_name,
                raw_frames=raw_frames,
                nodes=preparation.nodes,
                selected_nodes=selected_nodes,
                node_to_final_track=node_to_final_track,
                indexed_solutions=indexed_solutions,
                common_geometry_assignments=geometry_result.common_geometry_assignments,
            )
        }
        optimized_geometry_payloads[geometry_result.variant_name] = geometry_result.diagnostics
        geometry_assignments_json_path = geometry_assignments_path(output_root)
        reporting.write_json(
            geometry_assignments_json_path,
            {
                "common_geometry_mode": config.common_geometry_mode,
                "assignments": {
                    str(node_id): option_name
                    for node_id, option_name in sorted(geometry_result.common_geometry_assignments.items())
                },
                "diagnostics": geometry_result.diagnostics,
            },
        )
        LOGGER.info("Wrote geometry assignments to %s.", geometry_assignments_json_path)

    variant_names = tuple(tracked_masks_by_variant)
    render_manifest = build_render_manifest(
        config=config,
        variant_names=variant_names,
        geometry_assignments_filename=geometry_assignments_json_path.name if geometry_assignments_json_path is not None else None,
    )
    manifest_output_path = write_render_manifest(output_root, render_manifest)
    LOGGER.info("Wrote render manifest to %s.", manifest_output_path)
    return _finalize_consensus_outputs(
        config=config,
        raw_frames=raw_frames,
        indexed_solutions=indexed_solutions,
        preparation=preparation,
        output_root=output_root,
        source_names=source_names,
        lineage_rows=lineage_rows,
        tracked_masks_by_variant=tracked_masks_by_variant,
        premerge_json_path=premerge_json_path,
        premerge_text_path=premerge_text_path,
        diagnostics=FragmentDiagnostics(
            best_input_source=None,
            best_input_tra=None,
            graph_statistics={**preparation.graph_stats, **selected_graph_stats},
            oracle_metrics=preparation.oracle_metrics,
            variant_deltas_to_best_input={},
        ),
        write_outputs=True,
        optimized_geometry_payloads=optimized_geometry_payloads,
        manifest_output_path=manifest_output_path,
        geometry_assignments_json_path=geometry_assignments_json_path,
    )


def evaluate_saved_consensus_outputs(
    config: TrackingConfig,
    raw_frames: np.ndarray,
    solutions_by_source: dict[str, SavedTrackingSolution],
) -> ConsensusResult:
    if len(solutions_by_source) != 2:
        raise ValueError("Consensus evaluation currently expects exactly two saved source solutions.")

    source_names = tuple(config.consensus_sources)
    indexed_solutions = {name: build_solution_index(solution) for name, solution in solutions_by_source.items()}
    LOGGER.info("Recomputing consensus pre-merge metrics from saved source solutions.")
    preparation = prepare_consensus(config, indexed_solutions, scorers=None)
    output_root = projectio.resolve_consensus_output_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_render_manifest(output_root)
    variant_names = render_variant_names_from_manifest(manifest, source_names)

    premerge_json_path = output_root / "premerge_metrics.json"
    premerge_text_path = output_root / "premerge_metrics.txt"
    premerge_payload = {
        "agreement_metrics": preparation.input_metrics.agreement_metrics,
        "input_solution_metrics": preparation.input_metrics.input_solution_metrics,
        "common_tracklet_count": preparation.input_metrics.common_tracklet_count,
        "hypothesis_tracklet_count": preparation.input_metrics.hypothesis_tracklet_count,
        "graph_statistics": preparation.graph_stats,
        "candidate_oracle": preparation.oracle_metrics,
    }
    reporting.write_json(premerge_json_path, premerge_payload)
    reporting.write_text(premerge_text_path, reporting.format_metrics_report(premerge_payload))
    LOGGER.info("Wrote pre-merge metrics to %s and %s.", premerge_json_path, premerge_text_path)

    tracked_masks_by_variant: dict[str, np.ndarray] = {}
    lineage_rows: tuple[LineageRecord, ...] | None = None
    for variant_name in variant_names:
        variant_dir = output_root / variant_name
        saved_variant = projectio.load_saved_tracking_solution(variant_name, variant_dir, raw_frames)
        tracked_masks_by_variant[variant_name] = saved_variant.tracked_masks
        if lineage_rows is None:
            lineage_rows = saved_variant.lineage_rows
        elif lineage_rows != saved_variant.lineage_rows:
            raise ValueError(f"Saved consensus variant '{variant_name}' has lineage rows inconsistent with the other variants.")

    if lineage_rows is None:
        raise ValueError(f"No saved consensus variants were found under {output_root}.")

    geometry_assignments_json_path = saved_geometry_assignments_path(output_root, manifest)
    if manifest is None:
        manifest = build_render_manifest(
            config=config,
            variant_names=variant_names,
            geometry_assignments_filename=geometry_assignments_json_path.name if geometry_assignments_json_path is not None else None,
        )
    manifest_output_path = write_render_manifest(output_root, manifest)

    return _finalize_consensus_outputs(
        config=config,
        raw_frames=raw_frames,
        indexed_solutions=indexed_solutions,
        preparation=preparation,
        output_root=output_root,
        source_names=source_names,
        lineage_rows=lineage_rows,
        tracked_masks_by_variant=tracked_masks_by_variant,
        premerge_json_path=premerge_json_path,
        premerge_text_path=premerge_text_path,
        diagnostics=FragmentDiagnostics(
            best_input_source=None,
            best_input_tra=None,
            graph_statistics=preparation.graph_stats,
            oracle_metrics=preparation.oracle_metrics,
            variant_deltas_to_best_input={},
        ),
        write_outputs=False,
        optimized_geometry_payloads=load_saved_variant_geometry_metrics(output_root, variant_names),
        manifest_output_path=manifest_output_path,
        geometry_assignments_json_path=geometry_assignments_json_path,
    )


def _finalize_consensus_outputs(
    config: TrackingConfig,
    raw_frames: np.ndarray,
    indexed_solutions: dict[str, SolutionIndex],
    preparation: ConsensusPreparation,
    output_root: Path,
    source_names: tuple[str, str],
    lineage_rows: tuple[LineageRecord, ...],
    tracked_masks_by_variant: dict[str, np.ndarray],
    premerge_json_path: Path,
    premerge_text_path: Path,
    diagnostics: FragmentDiagnostics,
    write_outputs: bool,
    optimized_geometry_payloads: dict[str, dict[str, object]] | None = None,
    manifest_output_path: Path | None = None,
    geometry_assignments_json_path: Path | None = None,
) -> ConsensusResult:
    gt_index = _load_gt_solution_index(config, raw_frames)
    variant_evaluations: dict[str, VariantEvaluation] = {}
    comparison_payload: dict[str, dict[str, object]] = {}

    common_node_ids = {node.node_id for node in preparation.nodes if node.is_common_supported}
    selected_common_tracklets = 0
    best_input_source, best_input_tra = best_input_baseline(preparation.input_metrics.input_solution_metrics)
    variant_deltas: dict[str, dict[str, float]] = {}

    for variant_name, tracked_masks in tracked_masks_by_variant.items():
        variant_dir = output_root / variant_name
        if write_outputs:
            mask_paths, lineage_path = projectio.write_tracking_outputs(variant_dir, tracked_masks, lineage_rows)
        else:
            mask_paths = tuple(variant_dir / f"mask{frame_index:03d}.tif" for frame_index in range(len(tracked_masks)))
            lineage_path = variant_dir / "res_track.txt"
        metrics: dict[str, object] = {
            "summary_metrics": summarize_variant(
                variant_name=variant_name,
                tracked_masks=tracked_masks,
                lineage_rows=lineage_rows,
                common_tracklet_count=len(preparation.common_tracklets),
                selected_common_tracklets=len(common_node_ids),
            )
        }
        if optimized_geometry_payloads is not None and variant_name in optimized_geometry_payloads:
            metrics["optimized_geometry"] = dict(optimized_geometry_payloads[variant_name])
        if gt_index is not None:
            metrics["legacy_gt_metrics"] = evaluate_solution_against_gt(
                build_solution_index(
                    SavedTrackingSolution(
                        source_name=variant_name,
                        output_dir=variant_dir,
                        tracked_masks=tracked_masks,
                        lineage_rows=lineage_rows,
                        frames=tuple(
                            projectio.build_frame_objects(variant_name, frame_index, mask, raw_frames[frame_index])
                            for frame_index, mask in enumerate(tracked_masks)
                        ),
                        checkpoint=None,
                    )
                ),
                gt_index,
                GT_EVAL_IOU_THRESHOLD,
            )
        metrics["ctc_evaluation"] = evaluate_result_with_ctc(
            config.dataset_root / f"{config.track_sequence}_GT",
            variant_dir,
        )
        log_ctc_evaluation(f"Consensus variant '{variant_name}'", metrics["ctc_evaluation"])
        variant_deltas[variant_name] = ctc_metric_deltas(metrics.get("ctc_evaluation", {}), best_input_tra, best_input_source, preparation.input_metrics.input_solution_metrics)
        metrics_json_path = variant_dir / "metrics.json"
        metrics_text_path = variant_dir / "metrics.txt"
        reporting.write_json(metrics_json_path, metrics)
        reporting.write_text(metrics_text_path, reporting.format_metrics_report(metrics))
        comparison_payload[variant_name] = dict(metrics)
        variant_evaluations[variant_name] = VariantEvaluation(
            variant_name=variant_name,
            metrics=dict(metrics),
            output_dir=variant_dir,
            mask_paths=mask_paths,
            lineage_path=lineage_path,
            metrics_json_path=metrics_json_path,
            metrics_text_path=metrics_text_path,
        )
        LOGGER.info("Saved consensus variant '%s' metrics to %s.", variant_name, variant_dir)
        selected_common_tracklets = len(common_node_ids)

    comparison_json_path = output_root / "variant_comparison.json"
    comparison_text_path = output_root / "variant_comparison.txt"
    reporting.write_json(comparison_json_path, comparison_payload)
    reporting.write_text(comparison_text_path, reporting.format_metrics_report(comparison_payload))
    LOGGER.info("Wrote variant comparison metrics to %s and %s.", comparison_json_path, comparison_text_path)

    diagnostics_payload = {
        "best_input_source": best_input_source,
        "best_input_tra": best_input_tra,
        "graph_statistics": diagnostics.graph_statistics,
        "candidate_oracle": diagnostics.oracle_metrics,
        "variant_deltas_to_best_input": variant_deltas,
    }
    if optimized_geometry_payloads:
        diagnostics_payload["optimized_geometry"] = optimized_geometry_payloads
    diagnostics_json_path = output_root / "consensus_diagnostics.json"
    diagnostics_text_path = output_root / "consensus_diagnostics.txt"
    reporting.write_json(diagnostics_json_path, diagnostics_payload)
    reporting.write_text(diagnostics_text_path, reporting.format_metrics_report(diagnostics_payload))
    LOGGER.info("Wrote consensus diagnostics to %s and %s.", diagnostics_json_path, diagnostics_text_path)

    return ConsensusResult(
        selected_sources=source_names,
        lineage_rows=lineage_rows,
        output_dir=output_root,
        premerge_metrics_path=premerge_json_path,
        premerge_metrics_text_path=premerge_text_path,
        variant_comparison_path=comparison_json_path,
        variant_comparison_text_path=comparison_text_path,
        variant_evaluations=variant_evaluations,
        diagnostics_path=diagnostics_json_path,
        diagnostics_text_path=diagnostics_text_path,
        render_manifest_path=manifest_output_path,
        geometry_assignments_path=geometry_assignments_json_path,
    )


def _load_gt_solution_index(
    config: TrackingConfig,
    raw_frames: np.ndarray,
) -> SolutionIndex | None:
    try:
        gt_frames = projectio.load_gt_frame_objects(config.dataset_root, config.track_sequence, raw_frames)
        gt_rows = tuple(projectio.load_lineage_records(config.dataset_root, config.track_sequence).values())
        gt_solution = SavedTrackingSolution(
            source_name="gt",
            output_dir=config.dataset_root / f"{config.track_sequence}_GT" / "TRA",
            tracked_masks=np.stack([frame.label_image.astype(np.uint16) for frame in gt_frames]),
            lineage_rows=gt_rows,
            frames=tuple(gt_frames),
            checkpoint=None,
        )
        LOGGER.info("Loaded GT tracking data for consensus evaluation.")
        return build_solution_index(gt_solution)
    except ValueError:
        LOGGER.info("No GT tracking data found for %s; consensus evaluation will use summary metrics only.", config.track_sequence)
        return None


def build_solution_index(solution: SavedTrackingSolution) -> SolutionIndex:
    rows_by_track = {row.track_id: row for row in solution.lineage_rows}
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for row in solution.lineage_rows:
        if row.parent > 0:
            children_by_parent[row.parent].append(row.track_id)
    return SolutionIndex(
        solution=solution,
        rows_by_track=rows_by_track,
        children_by_parent={parent: tuple(sorted(children)) for parent, children in children_by_parent.items()},
    )


def prepare_consensus(
    config: TrackingConfig,
    indexed_solutions: dict[str, SolutionIndex],
    scorers: EventScorers | None,
) -> ConsensusPreparation:
    stage_total = 10
    source_name_1, source_name_2 = config.consensus_sources
    solution_1 = indexed_solutions[source_name_1]
    solution_2 = indexed_solutions[source_name_2]

    log_consensus_stage(1, stage_total, "Matching source solutions by frame.")
    matches_by_frame = match_solutions_by_frame(solution_1, solution_2, config.agreement_iou_threshold)
    matched_pairs = sum(len(frame_matches.pairs) for frame_matches in matches_by_frame)
    LOGGER.info("Consensus prep 1/%s complete: matched %s object pair(s) across %s frame(s).", stage_total, matched_pairs, len(matches_by_frame))

    log_consensus_stage(2, stage_total, "Splitting source trajectories into atomic fragments.")
    source_fragments = {
        source_name_1: build_source_fragments(solution_1, matches_by_frame, source_name_1),
        source_name_2: build_source_fragments(solution_2, matches_by_frame, source_name_2),
    }
    LOGGER.info(
        "Consensus prep 2/%s complete: %s fragment(s) for %s and %s fragment(s) for %s.",
        stage_total,
        len(source_fragments[source_name_1]),
        source_name_1,
        len(source_fragments[source_name_2]),
        source_name_2,
    )

    log_consensus_stage(3, stage_total, "Building common-supported fragments.")
    common_tracklets = build_common_tracklets(solution_1, solution_2, matches_by_frame, source_fragments)
    LOGGER.info("Consensus prep 3/%s complete: built %s common-supported fragment(s).", stage_total, len(common_tracklets))

    log_consensus_stage(4, stage_total, "Creating fragment graph nodes.")
    hypothesis_tracklets = tuple(source_fragments[source_name_1] + source_fragments[source_name_2])
    nodes = build_tracklet_nodes(common_tracklets, hypothesis_tracklets, indexed_solutions, matches_by_frame)
    LOGGER.info("Consensus prep 4/%s complete: created %s node(s).", stage_total, len(nodes))

    log_consensus_stage(5, stage_total, "Indexing node occupancy and frame lookups.")
    frame_index = build_node_frame_index(nodes, indexed_solutions, config.consensus_sources)
    LOGGER.info("Consensus prep 5/%s complete.", stage_total)

    log_consensus_stage(6, stage_total, "Building continuation candidates.")
    continuation_candidates = build_continuation_candidates(config, nodes, indexed_solutions)
    LOGGER.info("Consensus prep 6/%s complete: built %s continuation candidate(s).", stage_total, len(continuation_candidates))

    log_consensus_stage(7, stage_total, "Building spatial conflict pairs.")
    tolerated_pair_keys = {candidate.pair_key for candidate in continuation_candidates if candidate.is_tolerated_handoff}
    overlap_pairs = compute_overlap_pairs(nodes, frame_index, config.consensus_sources)
    hard_conflicts = tuple(sorted((pair for pair in overlap_pairs if pair not in tolerated_pair_keys), key=lambda pair: tuple(sorted(pair))))
    LOGGER.info(
        "Consensus prep 7/%s complete: %s overlap pair(s), %s hard conflict(s), %s tolerated handoff pair(s).",
        stage_total,
        len(overlap_pairs),
        len(hard_conflicts),
        len(tolerated_pair_keys),
    )

    log_consensus_stage(8, stage_total, "Building division candidates and fragment-graph summary.")
    division_candidates = build_division_candidates(nodes, continuation_candidates, overlap_pairs)
    graph_stats = summarize_fragment_graph(
        nodes=nodes,
        continuation_candidates=continuation_candidates,
        division_candidates=division_candidates,
        hard_conflicts=hard_conflicts,
    )
    LOGGER.info("Consensus prep 8/%s complete: built %s division candidate(s).", stage_total, len(division_candidates))

    log_consensus_stage(9, stage_total, "Evaluating saved input solutions against GT.")
    input_metrics = ConsensusMetrics(
        agreement_metrics=agreement_metrics(solution_1, solution_2, matches_by_frame, common_tracklets),
        input_solution_metrics=input_solution_metrics(indexed_solutions, config.dataset_root, config.track_sequence),
        common_tracklet_count=len(common_tracklets),
        hypothesis_tracklet_count=len(hypothesis_tracklets),
    )
    LOGGER.info("Consensus prep 9/%s complete.", stage_total)

    log_consensus_stage(10, stage_total, "Computing candidate-oracle coverage diagnostics.")
    oracle_metrics = candidate_oracle_metrics(
        config=config,
        indexed_solutions=indexed_solutions,
        nodes=nodes,
        continuation_candidates=continuation_candidates,
        division_candidates=division_candidates,
    )
    LOGGER.info("Consensus prep 10/%s complete.", stage_total)
    return ConsensusPreparation(
        common_tracklets=common_tracklets,
        hypothesis_tracklets=hypothesis_tracklets,
        nodes=nodes,
        matches_by_frame=matches_by_frame,
        input_metrics=input_metrics,
        continuation_candidates=continuation_candidates,
        division_candidates=division_candidates,
        hard_conflicts=hard_conflicts,
        overlap_pairs=tuple(sorted(overlap_pairs, key=lambda pair: tuple(sorted(pair)))),
        graph_stats=graph_stats,
        oracle_metrics=oracle_metrics,
    )


def match_solutions_by_frame(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    iou_threshold: float,
) -> tuple[FrameMatches, ...]:
    matches: list[FrameMatches] = []
    frame_pairs = zip(solution_1.solution.frames, solution_2.solution.frames, strict=True)
    for frame_1, frame_2 in progress_iter(
        frame_pairs,
        desc="Matching solutions by frame",
        total=len(solution_1.solution.frames),
        unit="frame",
    ):
        if frame_1.object_count == 0 or frame_2.object_count == 0:
            matches.append(FrameMatches(pairs=(), source_1_to_2={}, source_2_to_1={}, pair_iou={}))
            continue

        intersections = intersection_areas(frame_1, frame_2)
        cost_matrix = np.ones((frame_1.object_count, frame_2.object_count), dtype=float)
        pair_iou: dict[tuple[int, int], float] = {}
        for (left_track_id, right_track_id), intersection in intersections.items():
            left_index = frame_1.raw_label_to_index[left_track_id]
            right_index = frame_2.raw_label_to_index[right_track_id]
            union = frame_1.areas[left_index] + frame_2.areas[right_index] - intersection
            iou = 0.0 if union == 0 else float(intersection / union)
            cost_matrix[left_index, right_index] = 1.0 - iou
            pair_iou[(left_track_id, right_track_id)] = iou

        left_to_right: dict[int, int] = {}
        right_to_left: dict[int, int] = {}
        pairs: list[MatchedObjectPair] = []
        left_indices, right_indices = linear_sum_assignment(cost_matrix)
        for left_index, right_index in zip(left_indices, right_indices, strict=True):
            iou = 1.0 - float(cost_matrix[left_index, right_index])
            if iou < iou_threshold:
                continue
            left_track_id = frame_1.raw_label_ids[left_index]
            right_track_id = frame_2.raw_label_ids[right_index]
            left_to_right[left_track_id] = right_track_id
            right_to_left[right_track_id] = left_track_id
            pairs.append(
                MatchedObjectPair(
                    frame_index=frame_1.frame_index,
                    track_id_1=left_track_id,
                    track_id_2=right_track_id,
                    iou=iou,
                )
            )
        matches.append(
            FrameMatches(
                pairs=tuple(sorted(pairs, key=lambda pair: (pair.track_id_1, pair.track_id_2))),
                source_1_to_2=left_to_right,
                source_2_to_1=right_to_left,
                pair_iou=pair_iou,
            )
        )
    return tuple(matches)


def build_source_fragments(
    solution: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
    source_name: str,
) -> tuple[HypothesisTracklet, ...]:
    partner_lookup = [
        frame_matches.source_1_to_2 if source_name == solution.solution.source_name else frame_matches.source_2_to_1
        for frame_matches in matches_by_frame
    ]
    fragments: list[HypothesisTracklet] = []
    next_tracklet_id = 0
    lineage_rows = sorted(solution.solution.lineage_rows, key=lambda record: (record.begin, record.track_id))
    for row in progress_iter(
        lineage_rows,
        desc=f"Building source fragments [{source_name}]",
        total=len(lineage_rows),
        unit="track",
    ):
        boundaries = [row.begin]
        last_partner = partner_lookup[row.begin].get(row.track_id) if row.begin < len(partner_lookup) else None
        for frame_index in range(row.begin + 1, row.end + 1):
            current_partner = partner_lookup[frame_index].get(row.track_id) if frame_index < len(partner_lookup) else None
            if current_partner != last_partner:
                boundaries.append(frame_index)
            last_partner = current_partner
        boundaries.append(row.end + 1)
        for begin, stop in zip(boundaries, boundaries[1:]):
            end = stop - 1
            if begin > end:
                continue
            fragments.append(
                HypothesisTracklet(
                    tracklet_id=next_tracklet_id,
                    begin=begin,
                    end=end,
                    source_name=source_name,
                    source_track_id=row.track_id,
                )
            )
            next_tracklet_id += 1
    return tuple(fragments)


def build_common_tracklets(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
    source_fragments: dict[str, tuple[HypothesisTracklet, ...]],
) -> tuple[CommonTracklet, ...]:
    source_name_1 = solution_1.solution.source_name
    source_name_2 = solution_2.solution.source_name
    fragment_lookup = {
        (fragment.source_name, fragment.source_track_id, fragment.begin, fragment.end): fragment
        for fragments in source_fragments.values()
        for fragment in fragments
    }
    common_tracklets: list[CommonTracklet] = []
    seen_pairs: set[tuple[int, int, int, int]] = set()
    next_tracklet_id = 0
    left_fragments = source_fragments[source_name_1]
    for fragment in progress_iter(
        left_fragments,
        desc="Building common-supported fragments",
        total=len(left_fragments),
        unit="fragment",
    ):
        partner_track_id = matches_by_frame[fragment.begin].source_1_to_2.get(fragment.source_track_id)
        if partner_track_id is None:
            continue
        if not all(matches_by_frame[frame_index].source_1_to_2.get(fragment.source_track_id) == partner_track_id for frame_index in range(fragment.begin, fragment.end + 1)):
            continue
        partner_fragment = fragment_lookup.get((source_name_2, partner_track_id, fragment.begin, fragment.end))
        if partner_fragment is None:
            continue
        if not all(matches_by_frame[frame_index].source_2_to_1.get(partner_track_id) == fragment.source_track_id for frame_index in range(fragment.begin, fragment.end + 1)):
            continue
        signature = (fragment.begin, fragment.end, fragment.source_track_id, partner_track_id)
        if signature in seen_pairs:
            continue
        common_tracklets.append(
            CommonTracklet(
                tracklet_id=next_tracklet_id,
                begin=fragment.begin,
                end=fragment.end,
                source_names=(source_name_1, source_name_2),
                source_track_ids=(fragment.source_track_id, partner_track_id),
            )
        )
        seen_pairs.add(signature)
        next_tracklet_id += 1
    return tuple(common_tracklets)


def build_tracklet_nodes(
    common_tracklets: tuple[CommonTracklet, ...],
    hypothesis_tracklets: tuple[HypothesisTracklet, ...],
    indexed_solutions: dict[str, SolutionIndex],
    matches_by_frame: tuple[FrameMatches, ...],
) -> tuple[TrackletNode, ...]:
    nodes: list[TrackletNode] = []
    next_node_id = 0

    for tracklet in progress_iter(
        common_tracklets,
        desc="Creating common-supported nodes",
        total=len(common_tracklets),
        unit="node",
    ):
        solution_1 = indexed_solutions[tracklet.source_names[0]]
        solution_2 = indexed_solutions[tracklet.source_names[1]]
        mean_iou = average_match_iou(tracklet, matches_by_frame)
        start_iou = match_iou_at_frame(matches_by_frame[tracklet.begin], tracklet.source_track_ids)
        end_iou = match_iou_at_frame(matches_by_frame[tracklet.end], tracklet.source_track_ids)
        nodes.append(
            TrackletNode(
                node_id=next_node_id,
                begin=tracklet.begin,
                end=tracklet.end,
                fixed=False,
                kind="common_supported",
                source_name=None,
                source_track_id=None,
                source_names=tracklet.source_names,
                source_track_ids=tracklet.source_track_ids,
                start_stats=average_stats(
                    object_stats(solution_1, tracklet.begin, tracklet.source_track_ids[0]),
                    object_stats(solution_2, tracklet.begin, tracklet.source_track_ids[1]),
                ),
                end_stats=average_stats(
                    object_stats(solution_1, tracklet.end, tracklet.source_track_ids[0]),
                    object_stats(solution_2, tracklet.end, tracklet.source_track_ids[1]),
                ),
                mean_iou=mean_iou,
                agreement_strength=float((start_iou + end_iou) / 2.0),
            )
        )
        next_node_id += 1

    for tracklet in progress_iter(
        hypothesis_tracklets,
        desc="Creating source-specific nodes",
        total=len(hypothesis_tracklets),
        unit="node",
    ):
        solution = indexed_solutions[tracklet.source_name]
        nodes.append(
            TrackletNode(
                node_id=next_node_id,
                begin=tracklet.begin,
                end=tracklet.end,
                fixed=False,
                kind="source_specific",
                source_name=tracklet.source_name,
                source_track_id=tracklet.source_track_id,
                source_names=None,
                source_track_ids=None,
                start_stats=object_stats(solution, tracklet.begin, tracklet.source_track_id),
                end_stats=object_stats(solution, tracklet.end, tracklet.source_track_id),
                mean_iou=0.0,
                agreement_strength=0.0,
            )
        )
        next_node_id += 1

    return tuple(nodes)


def filter_conflicting_hypotheses(
    common_nodes: list[TrackletNode],
    hypothesis_nodes: list[TrackletNode],
    indexed_solutions: dict[str, SolutionIndex],
) -> tuple[TrackletNode, ...]:
    if not common_nodes:
        return tuple(hypothesis_nodes)

    fixed_union_occupancy = build_fixed_common_union_occupancy(common_nodes, indexed_solutions)
    kept: list[TrackletNode] = []
    for node in hypothesis_nodes:
        if overlaps_fixed_common_union(node, fixed_union_occupancy, indexed_solutions):
            continue
        kept.append(node)
    return tuple(kept)


def build_fixed_common_union_occupancy(
    common_nodes: list[TrackletNode],
    indexed_solutions: dict[str, SolutionIndex],
) -> list[np.ndarray]:
    sample_solution = next(iter(indexed_solutions.values())).solution
    frame_count = len(sample_solution.frames)
    frame_shape = sample_solution.frames[0].shape
    occupancy = [np.zeros(frame_shape, dtype=bool) for _ in range(frame_count)]
    for node in common_nodes:
        for frame_index in range(node.begin, node.end + 1):
            coords = node_variant_coords("union", node, frame_index, indexed_solutions)
            if coords.size == 0:
                continue
            occupancy[frame_index][coords[:, 0], coords[:, 1]] = True
    return occupancy


def overlaps_fixed_common_union(
    node: TrackletNode,
    fixed_union_occupancy: list[np.ndarray],
    indexed_solutions: dict[str, SolutionIndex],
) -> bool:
    for frame_index in range(node.begin, node.end + 1):
        coords = node_variant_coords("union", node, frame_index, indexed_solutions)
        if coords.size == 0:
            continue
        if np.any(fixed_union_occupancy[frame_index][coords[:, 0], coords[:, 1]]):
            return True
    return False


def build_node_frame_index(
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
    source_names: tuple[str, str],
) -> NodeFrameIndex:
    frame_count = len(next(iter(indexed_solutions.values())).solution.frames)
    source_nodes_by_frame_track = {
        source_name: tuple(defaultdict(int) for _ in range(frame_count))
        for source_name in source_names
    }
    common_nodes_by_frame_source_track = {
        source_name: tuple(defaultdict(set) for _ in range(frame_count))
        for source_name in source_names
    }
    common_nodes_by_frame: list[list[int]] = [[] for _ in range(frame_count)]

    for node in progress_iter(
        nodes,
        desc="Indexing fragment occupancy",
        total=len(nodes),
        unit="node",
    ):
        if node.is_common_supported:
            assert node.source_names is not None and node.source_track_ids is not None
            for frame_index in range(node.begin, node.end + 1):
                common_nodes_by_frame[frame_index].append(node.node_id)
                common_nodes_by_frame_source_track[node.source_names[0]][frame_index][node.source_track_ids[0]].add(node.node_id)
                common_nodes_by_frame_source_track[node.source_names[1]][frame_index][node.source_track_ids[1]].add(node.node_id)
        else:
            assert node.source_name is not None and node.source_track_id is not None
            for frame_index in range(node.begin, node.end + 1):
                source_nodes_by_frame_track[node.source_name][frame_index][node.source_track_id] = node.node_id

    cross_overlap_by_frame: list[dict[tuple[int, int], int]] = []
    left_solution = indexed_solutions[source_names[0]]
    right_solution = indexed_solutions[source_names[1]]
    cross_frame_pairs = zip(left_solution.solution.frames, right_solution.solution.frames, strict=True)
    for frame_left, frame_right in progress_iter(
        cross_frame_pairs,
        desc="Indexing cross-source overlap maps",
        total=len(left_solution.solution.frames),
        unit="frame",
    ):
        cross_overlap_by_frame.append(dict(intersection_areas(frame_left, frame_right)))

    return NodeFrameIndex(
        source_nodes_by_frame_track=source_nodes_by_frame_track,
        common_nodes_by_frame_source_track=common_nodes_by_frame_source_track,
        common_nodes_by_frame=tuple(tuple(node_ids) for node_ids in common_nodes_by_frame),
        cross_overlap_by_frame=tuple(cross_overlap_by_frame),
    )


def build_continuation_candidates(
    config: TrackingConfig,
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
) -> tuple[ContinuationCandidate, ...]:
    nodes_by_begin: dict[int, list[TrackletNode]] = defaultdict(list)
    candidates: list[ContinuationCandidate] = []
    for node in nodes:
        nodes_by_begin[node.begin].append(node)

    for parent in progress_iter(
        nodes,
        desc="Building continuation candidates",
        total=len(nodes),
        unit="node",
    ):
        for begin_frame in (parent.end, parent.end + 1):
            for child in nodes_by_begin.get(begin_frame, []):
                if parent.node_id == child.node_id:
                    continue
                if child.begin < parent.begin:
                    continue
                if centroid_distance(parent.end_stats, child.start_stats) > config.max_distance:
                    continue
                if child.begin == parent.end + 1:
                    candidates.append(
                        ContinuationCandidate(
                            parent_id=parent.node_id,
                            child_id=child.node_id,
                            shared_boundary_frame=None,
                            overlap_pixels=0,
                            overlap_fraction=0.0,
                        )
                    )
                    continue

                overlap_pixels, overlap_fraction = boundary_overlap_details(parent, child, parent.end, indexed_solutions)
                if overlap_pixels == 0:
                    continue
                smaller_area = min(parent.end_stats.area, child.start_stats.area)
                if (
                    overlap_pixels <= HANDOFF_OVERLAP_ABSOLUTE_PIXELS
                    and smaller_area > 0
                    and overlap_fraction <= HANDOFF_OVERLAP_RELATIVE_FRACTION
                ):
                    candidates.append(
                        ContinuationCandidate(
                            parent_id=parent.node_id,
                            child_id=child.node_id,
                            shared_boundary_frame=parent.end,
                            overlap_pixels=overlap_pixels,
                            overlap_fraction=overlap_fraction,
                        )
                    )
    unique_candidates = {
        (candidate.parent_id, candidate.child_id): candidate
        for candidate in candidates
    }
    return tuple(
        unique_candidates[key]
        for key in sorted(unique_candidates, key=lambda item: (item[0], item[1]))
    )


def build_division_candidates(
    nodes: tuple[TrackletNode, ...],
    continuation_candidates: tuple[ContinuationCandidate, ...],
    overlap_pairs: tuple[frozenset[int], ...],
) -> tuple[tuple[int, int, int], ...]:
    nodes_by_begin: dict[int, list[int]] = defaultdict(list)
    node_lookup = {node.node_id: node for node in nodes}
    overlap_pair_set = set(overlap_pairs)
    for node in nodes:
        nodes_by_begin[node.begin].append(node.node_id)

    continuation_set = {(candidate.parent_id, candidate.child_id) for candidate in continuation_candidates}
    division_candidates: list[tuple[int, int, int]] = []
    for parent in progress_iter(
        nodes,
        desc="Building division candidates",
        total=len(nodes),
        unit="node",
    ):
        child_ids = sorted(
            child_id
            for child_id in nodes_by_begin.get(parent.end + 1, [])
            if child_id != parent.node_id and (parent.node_id, child_id) in continuation_set
        )
        for child_id_1, child_id_2 in combinations(child_ids, 2):
            if frozenset((child_id_1, child_id_2)) in overlap_pair_set:
                continue
            division_candidates.append((parent.node_id, child_id_1, child_id_2))
    return tuple(division_candidates)


def compute_overlap_pairs(
    nodes: tuple[TrackletNode, ...],
    frame_index: NodeFrameIndex,
    source_names: tuple[str, str],
) -> tuple[frozenset[int], ...]:
    pair_set: set[frozenset[int]] = set()
    source_name_1, source_name_2 = source_names
    source_nodes_1 = frame_index.source_nodes_by_frame_track[source_name_1]
    source_nodes_2 = frame_index.source_nodes_by_frame_track[source_name_2]
    common_tracks_1 = frame_index.common_nodes_by_frame_source_track[source_name_1]
    common_tracks_2 = frame_index.common_nodes_by_frame_source_track[source_name_2]

    for frame_number, overlaps in progress_iter(
        enumerate(frame_index.cross_overlap_by_frame),
        desc="Building overlap conflicts",
        total=len(frame_index.cross_overlap_by_frame),
        unit="frame",
    ):
        for (track_id_1, track_id_2), _pixels in overlaps.items():
            node_id_1 = source_nodes_1[frame_number].get(track_id_1)
            node_id_2 = source_nodes_2[frame_number].get(track_id_2)
            if node_id_1 and node_id_2:
                pair_set.add(frozenset((node_id_1, node_id_2)))
            for common_node_id in common_tracks_1[frame_number].get(track_id_1, ()):
                if node_id_2 is not None:
                    pair_set.add(frozenset((common_node_id, node_id_2)))
            for common_node_id in common_tracks_2[frame_number].get(track_id_2, ()):
                if node_id_1 is not None:
                    pair_set.add(frozenset((common_node_id, node_id_1)))
            for common_left in common_tracks_1[frame_number].get(track_id_1, ()):
                for common_right in common_tracks_2[frame_number].get(track_id_2, ()):
                    if common_left != common_right:
                        pair_set.add(frozenset((common_left, common_right)))

        for track_id, node_ids in common_tracks_1[frame_number].items():
            source_node = source_nodes_1[frame_number].get(track_id)
            if source_node is not None:
                for common_node_id in node_ids:
                    pair_set.add(frozenset((source_node, common_node_id)))
            for left_id, right_id in combinations(sorted(node_ids), 2):
                pair_set.add(frozenset((left_id, right_id)))
        for track_id, node_ids in common_tracks_2[frame_number].items():
            source_node = source_nodes_2[frame_number].get(track_id)
            if source_node is not None:
                for common_node_id in node_ids:
                    pair_set.add(frozenset((source_node, common_node_id)))
            for left_id, right_id in combinations(sorted(node_ids), 2):
                pair_set.add(frozenset((left_id, right_id)))
    return tuple(sorted(pair_set, key=lambda pair: tuple(sorted(pair))))


def summarize_fragment_graph(
    nodes: tuple[TrackletNode, ...],
    continuation_candidates: tuple[ContinuationCandidate, ...],
    division_candidates: tuple[tuple[int, int, int], ...],
    hard_conflicts: tuple[frozenset[int], ...],
) -> dict[str, object]:
    source_counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.source_name is not None:
            source_counts[node.source_name] += 1
    tolerated = [candidate for candidate in continuation_candidates if candidate.is_tolerated_handoff]
    return {
        "common_supported_fragment_count": len([node for node in nodes if node.is_common_supported]),
        "source_specific_fragment_count": len([node for node in nodes if not node.is_common_supported]),
        "source_specific_fragment_count_by_source": dict(sorted(source_counts.items())),
        "hard_conflict_count": len(hard_conflicts),
        "tolerated_handoff_pair_count": len(tolerated),
        "move_edge_count": len(continuation_candidates),
        "division_edge_count": len(division_candidates),
    }


def solve_global_tracklet_ilp(
    config: TrackingConfig,
    preparation: ConsensusPreparation,
    indexed_solutions: dict[str, SolutionIndex],
    scorers: EventScorers,
) -> tuple[
    set[int],
    dict[int, tuple[str, int] | tuple[str, int, int] | None],
    dict[int, tuple[str, int] | tuple[str, int, int] | None],
    dict[str, object],
]:
    if gp is None or GRB is None or quicksum is None:
        raise RuntimeError("Gurobi is required to solve the consensus ILP but is not installed in this environment.")
    nodes = preparation.nodes
    if not nodes:
        return set(), {}, {}, {"selected_total_count": 0}

    node_lookup = {node.node_id: node for node in nodes}
    continuation_lookup = {(candidate.parent_id, candidate.child_id): candidate for candidate in preparation.continuation_candidates}

    model = gp.Model("PyTr2dConsensus")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    LOGGER.info(
        "Building consensus ILP model: nodes=%s, move_candidates=%s, division_candidates=%s, hard_conflicts=%s.",
        len(nodes),
        len(preparation.continuation_candidates),
        len(preparation.division_candidates),
        len(preparation.hard_conflicts),
    )

    activation_vars: dict[int, gp.Var] = {}
    appearance_vars: dict[int, gp.Var] = {}
    disappearance_vars: dict[int, gp.Var] = {}
    move_vars: dict[tuple[int, int], gp.Var] = {}
    division_vars: dict[tuple[int, int, int], gp.Var] = {}
    incoming_terms: dict[int, list[gp.Var]] = defaultdict(list)
    outgoing_terms: dict[int, list[gp.Var]] = defaultdict(list)
    internal_consistency_cache: dict[int, float] = {}
    objective_terms: list[gp.LinExpr] = []

    for node in progress_iter(
        nodes,
        desc="Consensus ILP build: activation variables",
        total=len(nodes),
        unit="node",
    ):
        activation = model.addVar(vtype=GRB.BINARY, name=f"act[{node.node_id}]")
        activation_vars[node.node_id] = activation
        objective_terms.append(config.segmentation_reward * node.frame_count * activation)
        objective_terms.append(fragment_bonus(node, activation, scorers, indexed_solutions, internal_consistency_cache))

    for node in progress_iter(
        nodes,
        desc="Consensus ILP build: boundary variables",
        total=len(nodes),
        unit="node",
    ):
        appearance = model.addVar(vtype=GRB.BINARY, name=f"app[{node.node_id}]")
        disappearance = model.addVar(vtype=GRB.BINARY, name=f"dis[{node.node_id}]")
        appearance_vars[node.node_id] = appearance
        disappearance_vars[node.node_id] = disappearance
        objective_terms.append(appearance_cost(scorers, node) * BOUNDARY_PENALTY_SCALE * appearance)
        objective_terms.append(disappearance_cost(scorers, node) * BOUNDARY_PENALTY_SCALE * disappearance)

    for candidate in progress_iter(
        preparation.continuation_candidates,
        desc="Consensus ILP build: move variables",
        total=len(preparation.continuation_candidates),
        unit="edge",
    ):
        parent = node_lookup[candidate.parent_id]
        child = node_lookup[candidate.child_id]
        variable = model.addVar(vtype=GRB.BINARY, name=f"move[{candidate.parent_id},{candidate.child_id}]")
        move_vars[(candidate.parent_id, candidate.child_id)] = variable
        outgoing_terms[candidate.parent_id].append(variable)
        incoming_terms[candidate.child_id].append(variable)
        objective_terms.append(move_cost(scorers, parent, child) * variable)

    for parent_id, child_id_1, child_id_2 in progress_iter(
        preparation.division_candidates,
        desc="Consensus ILP build: division variables",
        total=len(preparation.division_candidates),
        unit="edge",
    ):
        parent = node_lookup[parent_id]
        child_1 = node_lookup[child_id_1]
        child_2 = node_lookup[child_id_2]
        variable = model.addVar(vtype=GRB.BINARY, name=f"div[{parent_id},{child_id_1},{child_id_2}]")
        division_vars[(parent_id, child_id_1, child_id_2)] = variable
        outgoing_terms[parent_id].append(variable)
        incoming_terms[child_id_1].append(variable)
        incoming_terms[child_id_2].append(variable)
        objective_terms.append(division_cost(scorers, parent, child_1, child_2) * variable)

    constraint_count = 0
    for node in progress_iter(
        nodes,
        desc="Consensus ILP build: flow constraints",
        total=len(nodes),
        unit="node",
    ):
        target = activation_vars[node.node_id]
        model.addConstr(
            quicksum(incoming_terms[node.node_id]) + appearance_vars[node.node_id] == target,
            name=f"incoming[{node.node_id}]",
        )
        constraint_count += 1
        model.addConstr(
            quicksum(outgoing_terms[node.node_id]) + disappearance_vars[node.node_id] == target,
            name=f"outgoing[{node.node_id}]",
        )
        constraint_count += 1

    sorted_hard_conflicts = sorted((tuple(sorted(pair)) for pair in preparation.hard_conflicts))
    for left_id, right_id in progress_iter(
        sorted_hard_conflicts,
        desc="Consensus ILP build: hard-conflict constraints",
        total=len(sorted_hard_conflicts),
        unit="pair",
    ):
        model.addConstr(
            activation_vars[left_id] + activation_vars[right_id] <= 1,
            name=f"conflict[{left_id},{right_id}]",
        )
        constraint_count += 1

    tolerated_handoff_count = 0
    for candidate in progress_iter(
        preparation.continuation_candidates,
        desc="Consensus ILP build: handoff constraints",
        total=len(preparation.continuation_candidates),
        unit="edge",
    ):
        if not candidate.is_tolerated_handoff:
            continue
        tolerated_handoff_count += 1
        model.addConstr(
            activation_vars[candidate.parent_id] + activation_vars[candidate.child_id] - move_vars[(candidate.parent_id, candidate.child_id)] <= 1,
            name=f"handoff[{candidate.parent_id},{candidate.child_id}]",
        )
        constraint_count += 1

    LOGGER.info(
        "Consensus ILP: nodes=%s, common_supported=%s, source_specific=%s, move=%s, division=%s, constraints=%s, hard_conflicts=%s, tolerated_handoffs=%s.",
        len(nodes),
        len([node for node in nodes if node.is_common_supported]),
        len([node for node in nodes if not node.is_common_supported]),
        len(move_vars),
        len(division_vars),
        constraint_count,
        len(preparation.hard_conflicts),
        tolerated_handoff_count,
    )
    model.setObjective(quicksum(objective_terms), GRB.MINIMIZE)
    LOGGER.info("Starting Gurobi optimization for consensus ILP.")
    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi failed to find a usable consensus solution. Status code: {model.Status}")

    selected_nodes = {node_id for node_id, variable in activation_vars.items() if variable.X > 0.5}

    incoming_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None] = {}
    outgoing_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None] = {}
    for node_id in selected_nodes:
        incoming_choice[node_id] = None
        outgoing_choice[node_id] = None
    for (parent_id, child_id), variable in move_vars.items():
        if variable.X > 0.5:
            outgoing_choice[parent_id] = ("move", child_id)
            incoming_choice[child_id] = ("move", parent_id)
    for (parent_id, child_id_1, child_id_2), variable in division_vars.items():
        if variable.X > 0.5:
            outgoing_choice[parent_id] = ("division", child_id_1, child_id_2)
            incoming_choice[child_id_1] = ("division", parent_id, child_id_2)
            incoming_choice[child_id_2] = ("division", parent_id, child_id_1)
    for node_id, variable in appearance_vars.items():
        if node_id in selected_nodes and variable.X > 0.5:
            incoming_choice[node_id] = None
    for node_id, variable in disappearance_vars.items():
        if node_id in selected_nodes and variable.X > 0.5:
            outgoing_choice[node_id] = None

    selected_source_counts: dict[str, int] = defaultdict(int)
    selected_common_count = 0
    cross_source_continuations = 0
    for node_id in selected_nodes:
        node = node_lookup[node_id]
        if node.is_common_supported:
            selected_common_count += 1
        elif node.source_name is not None:
            selected_source_counts[node.source_name] += 1
    for (parent_id, child_id), variable in move_vars.items():
        if variable.X <= 0.5:
            continue
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        if dominant_source(parent) != dominant_source(child):
            cross_source_continuations += 1

    LOGGER.info("Consensus ILP selected %s fragment node(s).", len(selected_nodes))
    return selected_nodes, incoming_choice, outgoing_choice, {
        "selected_total_count": len(selected_nodes),
        "selected_common_supported_count": selected_common_count,
        "selected_source_specific_count_by_source": dict(sorted(selected_source_counts.items())),
        "selected_cross_source_continuation_count": cross_source_continuations,
    }


def fixed_geometry_option(node: TrackletNode) -> str:
    if node.source_name is None:
        raise ValueError(f"Node {node.node_id} does not have a fixed source geometry.")
    return node.source_name


def node_geometry_options(
    node: TrackletNode,
    source_names: tuple[str, str],
) -> tuple[str, ...]:
    if node.is_common_supported:
        return geometry_option_names(source_names)
    return (fixed_geometry_option(node),)


def selected_temporal_relations(
    selected_nodes: set[int],
    outgoing_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None],
) -> tuple[tuple[str, int, int, float], ...]:
    relations: list[tuple[str, int, int, float]] = []
    for parent_id in sorted(selected_nodes):
        outgoing = outgoing_choice.get(parent_id)
        if outgoing is None:
            continue
        if outgoing[0] == "move":
            relations.append(("move", parent_id, int(outgoing[1]), 1.0))
            continue
        relations.append(("division", parent_id, int(outgoing[1]), 0.5))
        relations.append(("division", parent_id, int(outgoing[2]), 0.5))
    return tuple(relations)


def solve_two_stage_geometry_ilp(
    config: TrackingConfig,
    preparation: ConsensusPreparation,
    indexed_solutions: dict[str, SolutionIndex],
    selected_nodes: set[int],
    incoming_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None],
    outgoing_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None],
) -> GeometryOptimizationResult:
    if gp is None or GRB is None or quicksum is None:
        raise RuntimeError("Gurobi is required to solve the two-stage geometry ILP but is not installed in this environment.")

    source_names = tuple(config.consensus_sources)
    node_lookup = {node.node_id: node for node in preparation.nodes}
    selected_common_ids = sorted(node.node_id for node in preparation.nodes if node.node_id in selected_nodes and node.is_common_supported)
    if not selected_common_ids:
        return GeometryOptimizationResult(
            variant_name=optimized_variant_name(TWO_STAGE_GEOMETRY_MODE),
            common_geometry_assignments={},
            diagnostics=geometry_diagnostics(
                common_geometry_assignments={},
                node_lookup=node_lookup,
                source_names=source_names,
                indexed_solutions=indexed_solutions,
                selected_nodes=selected_nodes,
                temporal_relations=selected_temporal_relations(selected_nodes, outgoing_choice),
                same_frame_pairs=(),
                source_weight=config.geometry_source_weight,
                temporal_weight=config.geometry_temporal_overlap_weight,
                neighbor_radius=config.geometry_neighbor_radius,
            ),
        )

    selected_source_ids = {
        node.node_id
        for node in preparation.nodes
        if node.node_id in selected_nodes and not node.is_common_supported
    }
    LOGGER.info(
        "Two-stage geometry setup: selected_common=%s, selected_source=%s.",
        len(selected_common_ids),
        len(selected_source_ids),
    )
    geometry_cache = GeometryQueryCache()
    same_frame_pairs = find_same_frame_neighbor_pairs(
        nodes=preparation.nodes,
        indexed_solutions=indexed_solutions,
        source_names=source_names,
        radius=config.geometry_neighbor_radius,
        selected_node_ids=selected_nodes,
        cache=geometry_cache,
        progress_desc="Finding two-stage same-frame geometry neighbors",
    )
    frame_count = len(next(iter(indexed_solutions.values())).solution.frames)
    common_source_candidates, common_common_candidates = build_common_overlap_candidate_pairs(
        nodes=preparation.nodes,
        frame_count=frame_count,
        candidate_node_ids=selected_nodes,
    )
    common_source_candidate_count = sum(len(source_ids) for source_ids in common_source_candidates.values())
    LOGGER.info(
        "Two-stage geometry candidate pruning: common/source=%s, common/common=%s, same-frame neighbor pairs=%s.",
        common_source_candidate_count,
        len(common_common_candidates),
        len(same_frame_pairs),
    )
    temporal_relations = selected_temporal_relations(selected_nodes, outgoing_choice)

    model = gp.Model("PyTr2dGeometryTwoStage")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    geometry_vars: dict[tuple[int, str], gp.Var] = {}
    objective_terms: list[gp.LinExpr] = []
    for node_id in selected_common_ids:
        options = geometry_option_names(source_names)
        for option_name in options:
            geometry_vars[(node_id, option_name)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"geom[{node_id},{option_name}]",
            )
        model.addConstr(
            quicksum(geometry_vars[(node_id, option_name)] for option_name in options) == 1,
            name=f"geom_one[{node_id}]",
        )

    forbidden_option_count = 0
    for node_id in progress_iter(
        selected_common_ids,
        desc="Building two-stage source-conflict forbids",
        total=len(selected_common_ids),
        unit="common fragment",
    ):
        node = node_lookup[node_id]
        candidate_source_ids = common_source_candidates.get(node_id, ())
        if not candidate_source_ids:
            continue
        for option_name in geometry_option_names(source_names):
            if any(
                option_pair_overlaps_any_shared_frame(
                    node,
                    option_name,
                    node_lookup[source_node_id],
                    fixed_geometry_option(node_lookup[source_node_id]),
                    indexed_solutions,
                    cache=geometry_cache,
                )
                for source_node_id in candidate_source_ids
            ):
                model.addConstr(geometry_vars[(node_id, option_name)] == 0, name=f"geom_forbid_source[{node_id},{option_name}]")
                forbidden_option_count += 1

    LOGGER.info("Two-stage geometry forbids added for %s common option(s).", forbidden_option_count)

    common_conflict_constraint_count = 0
    for left_id, right_id in progress_iter(
        common_common_candidates,
        desc="Building two-stage common/common conflicts",
        total=len(common_common_candidates),
        unit="pair",
    ):
        left_node = node_lookup[left_id]
        right_node = node_lookup[right_id]
        for left_option in geometry_option_names(source_names):
            for right_option in geometry_option_names(source_names):
                if option_pair_overlaps_any_shared_frame(
                    left_node,
                    left_option,
                    right_node,
                    right_option,
                    indexed_solutions,
                    cache=geometry_cache,
                ):
                    model.addConstr(
                        geometry_vars[(left_id, left_option)] + geometry_vars[(right_id, right_option)] <= 1,
                        name=f"geom_conflict[{left_id},{left_option},{right_id},{right_option}]",
                    )
                    common_conflict_constraint_count += 1

    LOGGER.info("Two-stage geometry added %s common/common conflict constraint(s).", common_conflict_constraint_count)

    for _kind, parent_id, child_id, relation_weight in temporal_relations:
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        if not (parent.is_common_supported or child.is_common_supported):
            continue
        if parent.is_common_supported and child.is_common_supported:
            for parent_option in geometry_option_names(source_names):
                for child_option in geometry_option_names(source_names):
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"geom_temporal[{parent_id},{parent_option},{child_id},{child_option}]",
                    )
                    model.addConstr(variable <= geometry_vars[(parent_id, parent_option)])
                    model.addConstr(variable <= geometry_vars[(child_id, child_option)])
                    model.addConstr(
                        variable >= geometry_vars[(parent_id, parent_option)] + geometry_vars[(child_id, child_option)] - 1
                    )
                    objective_terms.append(
                        relation_weight
                        * geometry_temporal_score(
                            parent=parent,
                            parent_option=parent_option,
                            child=child,
                            child_option=child_option,
                            source_names=source_names,
                            indexed_solutions=indexed_solutions,
                            source_weight=config.geometry_source_weight,
                            temporal_weight=config.geometry_temporal_overlap_weight,
                            cache=geometry_cache,
                        )
                        * variable
                    )
        else:
            common_node = parent if parent.is_common_supported else child
            fixed_node = child if parent.is_common_supported else parent
            fixed_option = fixed_geometry_option(fixed_node)
            for common_option in geometry_option_names(source_names):
                objective_terms.append(
                    relation_weight
                    * geometry_temporal_score(
                        parent=parent if parent.is_common_supported else fixed_node,
                        parent_option=common_option if parent.is_common_supported else fixed_option,
                        child=child if child.is_common_supported else fixed_node,
                        child_option=common_option if child.is_common_supported else fixed_option,
                        source_names=source_names,
                        indexed_solutions=indexed_solutions,
                        source_weight=config.geometry_source_weight,
                        temporal_weight=config.geometry_temporal_overlap_weight,
                        cache=geometry_cache,
                    )
                    * geometry_vars[(common_node.node_id, common_option)]
                )

    for left_id, right_id in same_frame_pairs:
        left_node = node_lookup[left_id]
        right_node = node_lookup[right_id]
        if left_node.is_common_supported and right_node.is_common_supported:
            for left_option in geometry_option_names(source_names):
                for right_option in geometry_option_names(source_names):
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"geom_neighbor[{left_id},{left_option},{right_id},{right_option}]",
                    )
                    model.addConstr(variable <= geometry_vars[(left_id, left_option)])
                    model.addConstr(variable <= geometry_vars[(right_id, right_option)])
                    model.addConstr(
                        variable >= geometry_vars[(left_id, left_option)] + geometry_vars[(right_id, right_option)] - 1
                    )
                    objective_terms.append(
                        geometry_same_frame_score(
                            left_option=left_option,
                            right_option=right_option,
                            source_names=source_names,
                            source_weight=config.geometry_source_weight,
                        )
                        * variable
                    )
        else:
            common_node = left_node if left_node.is_common_supported else right_node
            fixed_node = right_node if left_node.is_common_supported else left_node
            fixed_option = fixed_geometry_option(fixed_node)
            for common_option in geometry_option_names(source_names):
                objective_terms.append(
                    geometry_same_frame_score(
                        left_option=common_option if left_node.is_common_supported else fixed_option,
                        right_option=common_option if right_node.is_common_supported else fixed_option,
                        source_names=source_names,
                        source_weight=config.geometry_source_weight,
                    )
                    * geometry_vars[(common_node.node_id, common_option)]
                )

    model.setObjective(quicksum(objective_terms), GRB.MAXIMIZE)
    LOGGER.info(
        "Starting Gurobi optimization for two-stage geometry ILP with %s selected common node(s).",
        len(selected_common_ids),
    )
    model.optimize()
    _assert_usable_status(model)

    assignments = {
        node_id: option_name
        for (node_id, option_name), variable in geometry_vars.items()
        if variable.X > 0.5
    }
    diagnostics = geometry_diagnostics(
        common_geometry_assignments=assignments,
        node_lookup=node_lookup,
        source_names=source_names,
        indexed_solutions=indexed_solutions,
        selected_nodes=selected_nodes,
        temporal_relations=temporal_relations,
        same_frame_pairs=same_frame_pairs,
        source_weight=config.geometry_source_weight,
        temporal_weight=config.geometry_temporal_overlap_weight,
        neighbor_radius=config.geometry_neighbor_radius,
        cache=geometry_cache,
    )
    return GeometryOptimizationResult(
        variant_name=optimized_variant_name(TWO_STAGE_GEOMETRY_MODE),
        common_geometry_assignments=assignments,
        diagnostics=diagnostics,
    )


def solve_joint_tracklet_geometry_ilp(
    config: TrackingConfig,
    preparation: ConsensusPreparation,
    indexed_solutions: dict[str, SolutionIndex],
    scorers: EventScorers,
) -> tuple[
    set[int],
    dict[int, tuple[str, int] | tuple[str, int, int] | None],
    dict[int, tuple[str, int] | tuple[str, int, int] | None],
    dict[str, object],
    dict[int, str],
    dict[str, object],
]:
    if gp is None or GRB is None or quicksum is None:
        raise RuntimeError("Gurobi is required to solve the joint geometry ILP but is not installed in this environment.")

    source_names = tuple(config.consensus_sources)
    nodes = preparation.nodes
    if not nodes:
        return set(), {}, {}, {"selected_total_count": 0}, {}, {}

    node_lookup = {node.node_id: node for node in nodes}
    option_names = geometry_option_names(source_names)
    common_node_ids = [node.node_id for node in nodes if node.is_common_supported]
    source_node_ids = [node.node_id for node in nodes if not node.is_common_supported]
    source_specific_hard_conflicts = tuple(
        pair
        for pair in preparation.hard_conflicts
        if all(not node_lookup[node_id].is_common_supported for node_id in pair)
    )
    frame_count = len(next(iter(indexed_solutions.values())).solution.frames)
    geometry_cache = GeometryQueryCache()
    LOGGER.info(
        "Joint geometry setup: nodes=%s, common_supported=%s, source_specific=%s.",
        len(nodes),
        len(common_node_ids),
        len(source_node_ids),
    )
    same_frame_pairs = find_same_frame_neighbor_pairs(
        nodes=nodes,
        indexed_solutions=indexed_solutions,
        source_names=source_names,
        radius=config.geometry_neighbor_radius,
        selected_node_ids=None,
        cache=geometry_cache,
        progress_desc="Finding joint same-frame geometry neighbors",
    )
    common_source_candidates, common_common_candidates = build_common_overlap_candidate_pairs(
        nodes=nodes,
        frame_count=frame_count,
        candidate_node_ids=None,
    )
    common_source_candidate_count = sum(len(source_ids) for source_ids in common_source_candidates.values())
    LOGGER.info(
        "Joint geometry candidate pruning: common/source=%s, common/common=%s, same-frame neighbor pairs=%s.",
        common_source_candidate_count,
        len(common_common_candidates),
        len(same_frame_pairs),
    )

    model = gp.Model("PyTr2dConsensusJoint")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    activation_vars: dict[int, gp.Var] = {}
    appearance_vars: dict[int, gp.Var] = {}
    disappearance_vars: dict[int, gp.Var] = {}
    move_vars: dict[tuple[int, int], gp.Var] = {}
    division_vars: dict[tuple[int, int, int], gp.Var] = {}
    geometry_vars: dict[tuple[int, str], gp.Var] = {}
    incoming_terms: dict[int, list[gp.Var]] = defaultdict(list)
    outgoing_terms: dict[int, list[gp.Var]] = defaultdict(list)
    internal_consistency_cache: dict[int, float] = {}
    objective_terms: list[gp.LinExpr] = []

    for node in progress_iter(
        nodes,
        desc="Joint ILP build: activation and geometry variables",
        total=len(nodes),
        unit="node",
    ):
        activation = model.addVar(vtype=GRB.BINARY, name=f"act[{node.node_id}]")
        activation_vars[node.node_id] = activation
        objective_terms.append(config.segmentation_reward * node.frame_count * activation)
        objective_terms.append(fragment_bonus(node, activation, scorers, indexed_solutions, internal_consistency_cache))
        if node.is_common_supported:
            for option_name in option_names:
                geometry_vars[(node.node_id, option_name)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"geom[{node.node_id},{option_name}]",
                )
            model.addConstr(
                quicksum(geometry_vars[(node.node_id, option_name)] for option_name in option_names) == activation,
                name=f"geom_one[{node.node_id}]",
            )

    for node in progress_iter(
        nodes,
        desc="Joint ILP build: boundary variables",
        total=len(nodes),
        unit="node",
    ):
        appearance = model.addVar(vtype=GRB.BINARY, name=f"app[{node.node_id}]")
        disappearance = model.addVar(vtype=GRB.BINARY, name=f"dis[{node.node_id}]")
        appearance_vars[node.node_id] = appearance
        disappearance_vars[node.node_id] = disappearance
        objective_terms.append(appearance_cost(scorers, node) * BOUNDARY_PENALTY_SCALE * appearance)
        objective_terms.append(disappearance_cost(scorers, node) * BOUNDARY_PENALTY_SCALE * disappearance)

    for candidate in progress_iter(
        preparation.continuation_candidates,
        desc="Joint ILP build: move variables",
        total=len(preparation.continuation_candidates),
        unit="edge",
    ):
        parent = node_lookup[candidate.parent_id]
        child = node_lookup[candidate.child_id]
        variable = model.addVar(vtype=GRB.BINARY, name=f"move[{candidate.parent_id},{candidate.child_id}]")
        move_vars[(candidate.parent_id, candidate.child_id)] = variable
        outgoing_terms[candidate.parent_id].append(variable)
        incoming_terms[candidate.child_id].append(variable)
        objective_terms.append(move_cost(scorers, parent, child) * variable)

    for parent_id, child_id_1, child_id_2 in progress_iter(
        preparation.division_candidates,
        desc="Joint ILP build: division variables",
        total=len(preparation.division_candidates),
        unit="edge",
    ):
        parent = node_lookup[parent_id]
        child_1 = node_lookup[child_id_1]
        child_2 = node_lookup[child_id_2]
        variable = model.addVar(vtype=GRB.BINARY, name=f"div[{parent_id},{child_id_1},{child_id_2}]")
        division_vars[(parent_id, child_id_1, child_id_2)] = variable
        outgoing_terms[parent_id].append(variable)
        incoming_terms[child_id_1].append(variable)
        incoming_terms[child_id_2].append(variable)
        objective_terms.append(division_cost(scorers, parent, child_1, child_2) * variable)

    constraint_count = 0
    for node in progress_iter(
        nodes,
        desc="Joint ILP build: flow constraints",
        total=len(nodes),
        unit="node",
    ):
        target = activation_vars[node.node_id]
        model.addConstr(quicksum(incoming_terms[node.node_id]) + appearance_vars[node.node_id] == target, name=f"incoming[{node.node_id}]")
        model.addConstr(quicksum(outgoing_terms[node.node_id]) + disappearance_vars[node.node_id] == target, name=f"outgoing[{node.node_id}]")
        constraint_count += 2

    sorted_hard_conflicts = sorted((tuple(sorted(pair)) for pair in source_specific_hard_conflicts))
    for left_id, right_id in progress_iter(
        sorted_hard_conflicts,
        desc="Joint ILP build: source-specific hard conflicts",
        total=len(sorted_hard_conflicts),
        unit="pair",
    ):
        model.addConstr(
            activation_vars[left_id] + activation_vars[right_id] <= 1,
            name=f"conflict_static[{left_id},{right_id}]",
        )
        constraint_count += 1

    for candidate in progress_iter(
        preparation.continuation_candidates,
        desc="Joint ILP build: tolerated handoff constraints",
        total=len(preparation.continuation_candidates),
        unit="edge",
    ):
        if not candidate.is_tolerated_handoff:
            continue
        model.addConstr(
            activation_vars[candidate.parent_id] + activation_vars[candidate.child_id] - move_vars[(candidate.parent_id, candidate.child_id)] <= 1,
            name=f"handoff[{candidate.parent_id},{candidate.child_id}]",
        )
        constraint_count += 1

    joint_source_conflict_count = 0
    for common_node_id in progress_iter(
        common_node_ids,
        desc="Joint ILP build: common/source geometry conflicts",
        total=len(common_node_ids),
        unit="common fragment",
    ):
        common_node = node_lookup[common_node_id]
        candidate_source_ids = common_source_candidates.get(common_node_id, ())
        if not candidate_source_ids:
            continue
        for option_name in option_names:
            for source_node_id in candidate_source_ids:
                source_node = node_lookup[source_node_id]
                if option_pair_overlaps_any_shared_frame(
                    common_node,
                    option_name,
                    source_node,
                    fixed_geometry_option(source_node),
                    indexed_solutions,
                    cache=geometry_cache,
                ):
                    model.addConstr(
                        geometry_vars[(common_node_id, option_name)] + activation_vars[source_node_id] <= 1,
                        name=f"geom_source_conflict[{common_node_id},{option_name},{source_node_id}]",
                    )
                    constraint_count += 1
                    joint_source_conflict_count += 1

    LOGGER.info("Joint ILP build added %s common/source geometry conflict constraint(s).", joint_source_conflict_count)

    joint_common_conflict_count = 0
    for left_id, right_id in progress_iter(
        common_common_candidates,
        desc="Joint ILP build: common/common geometry conflicts",
        total=len(common_common_candidates),
        unit="pair",
    ):
        left_node = node_lookup[left_id]
        right_node = node_lookup[right_id]
        for left_option in option_names:
            for right_option in option_names:
                if option_pair_overlaps_any_shared_frame(
                    left_node,
                    left_option,
                    right_node,
                    right_option,
                    indexed_solutions,
                    cache=geometry_cache,
                ):
                    model.addConstr(
                        geometry_vars[(left_id, left_option)] + geometry_vars[(right_id, right_option)] <= 1,
                        name=f"geom_conflict[{left_id},{left_option},{right_id},{right_option}]",
                    )
                    constraint_count += 1
                    joint_common_conflict_count += 1

    LOGGER.info("Joint ILP build added %s common/common geometry conflict constraint(s).", joint_common_conflict_count)

    for (parent_id, child_id), move_var in progress_iter(
        move_vars.items(),
        desc="Joint ILP build: temporal geometry auxiliaries",
        total=len(move_vars),
        unit="edge",
    ):
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        if not (parent.is_common_supported or child.is_common_supported):
            continue
        if parent.is_common_supported and child.is_common_supported:
            for parent_option in option_names:
                for child_option in option_names:
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"geom_move[{parent_id},{parent_option},{child_id},{child_option}]",
                    )
                    model.addConstr(variable <= move_var)
                    model.addConstr(variable <= geometry_vars[(parent_id, parent_option)])
                    model.addConstr(variable <= geometry_vars[(child_id, child_option)])
                    model.addConstr(
                        variable >= move_var + geometry_vars[(parent_id, parent_option)] + geometry_vars[(child_id, child_option)] - 2
                    )
                    objective_terms.append(
                        -geometry_temporal_score(
                            parent=parent,
                            parent_option=parent_option,
                            child=child,
                            child_option=child_option,
                            source_names=source_names,
                            indexed_solutions=indexed_solutions,
                            source_weight=config.geometry_source_weight,
                            temporal_weight=config.geometry_temporal_overlap_weight,
                            cache=geometry_cache,
                        )
                        * variable
                    )
        else:
            common_node = parent if parent.is_common_supported else child
            fixed_node = child if parent.is_common_supported else parent
            fixed_option = fixed_geometry_option(fixed_node)
            for common_option in option_names:
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"geom_move_fixed[{parent_id},{child_id},{common_option}]",
                )
                model.addConstr(variable <= move_var)
                model.addConstr(variable <= geometry_vars[(common_node.node_id, common_option)])
                model.addConstr(variable >= move_var + geometry_vars[(common_node.node_id, common_option)] - 1)
                objective_terms.append(
                    -geometry_temporal_score(
                        parent=parent if parent.is_common_supported else fixed_node,
                        parent_option=common_option if parent.is_common_supported else fixed_option,
                        child=child if child.is_common_supported else fixed_node,
                        child_option=common_option if child.is_common_supported else fixed_option,
                        source_names=source_names,
                        indexed_solutions=indexed_solutions,
                        source_weight=config.geometry_source_weight,
                        temporal_weight=config.geometry_temporal_overlap_weight,
                        cache=geometry_cache,
                    )
                    * variable
                )

    for (parent_id, child_id_1, child_id_2), division_var in progress_iter(
        division_vars.items(),
        desc="Joint ILP build: division geometry auxiliaries",
        total=len(division_vars),
        unit="edge",
    ):
        parent = node_lookup[parent_id]
        child_1 = node_lookup[child_id_1]
        child_2 = node_lookup[child_id_2]
        for child in (child_1, child_2):
            if not (parent.is_common_supported or child.is_common_supported):
                continue
            if parent.is_common_supported and child.is_common_supported:
                for parent_option in option_names:
                    for child_option in option_names:
                        variable = model.addVar(
                            vtype=GRB.BINARY,
                            name=f"geom_div[{parent_id},{parent_option},{child.node_id},{child_option}]",
                        )
                        model.addConstr(variable <= division_var)
                        model.addConstr(variable <= geometry_vars[(parent_id, parent_option)])
                        model.addConstr(variable <= geometry_vars[(child.node_id, child_option)])
                        model.addConstr(
                            variable >= division_var + geometry_vars[(parent_id, parent_option)] + geometry_vars[(child.node_id, child_option)] - 2
                        )
                        objective_terms.append(
                            -0.5
                            * geometry_temporal_score(
                                parent=parent,
                                parent_option=parent_option,
                                child=child,
                                child_option=child_option,
                                source_names=source_names,
                                indexed_solutions=indexed_solutions,
                                source_weight=config.geometry_source_weight,
                                temporal_weight=config.geometry_temporal_overlap_weight,
                                cache=geometry_cache,
                            )
                            * variable
                        )
            else:
                common_node = parent if parent.is_common_supported else child
                fixed_node = child if parent.is_common_supported else parent
                fixed_option = fixed_geometry_option(fixed_node)
                for common_option in option_names:
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"geom_div_fixed[{parent_id},{child.node_id},{common_option}]",
                    )
                    model.addConstr(variable <= division_var)
                    model.addConstr(variable <= geometry_vars[(common_node.node_id, common_option)])
                    model.addConstr(variable >= division_var + geometry_vars[(common_node.node_id, common_option)] - 1)
                    objective_terms.append(
                        -0.5
                        * geometry_temporal_score(
                            parent=parent if parent.is_common_supported else fixed_node,
                            parent_option=common_option if parent.is_common_supported else fixed_option,
                            child=child if child.is_common_supported else fixed_node,
                            child_option=common_option if child.is_common_supported else fixed_option,
                            source_names=source_names,
                            indexed_solutions=indexed_solutions,
                            source_weight=config.geometry_source_weight,
                            temporal_weight=config.geometry_temporal_overlap_weight,
                            cache=geometry_cache,
                        )
                        * variable
                    )

    for left_id, right_id in progress_iter(
        same_frame_pairs,
        desc="Joint ILP build: same-frame geometry auxiliaries",
        total=len(same_frame_pairs),
        unit="pair",
    ):
        left_node = node_lookup[left_id]
        right_node = node_lookup[right_id]
        if left_node.is_common_supported and right_node.is_common_supported:
            for left_option in option_names:
                for right_option in option_names:
                    variable = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"geom_neighbor[{left_id},{left_option},{right_id},{right_option}]",
                    )
                    model.addConstr(variable <= geometry_vars[(left_id, left_option)])
                    model.addConstr(variable <= geometry_vars[(right_id, right_option)])
                    model.addConstr(
                        variable >= geometry_vars[(left_id, left_option)] + geometry_vars[(right_id, right_option)] - 1
                    )
                    objective_terms.append(
                        -geometry_same_frame_score(
                            left_option=left_option,
                            right_option=right_option,
                            source_names=source_names,
                            source_weight=config.geometry_source_weight,
                        )
                        * variable
                    )
        else:
            common_node = left_node if left_node.is_common_supported else right_node
            fixed_node = right_node if left_node.is_common_supported else left_node
            fixed_option = fixed_geometry_option(fixed_node)
            for common_option in option_names:
                variable = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"geom_neighbor_fixed[{left_id},{right_id},{common_option}]",
                )
                model.addConstr(variable <= activation_vars[fixed_node.node_id])
                model.addConstr(variable <= geometry_vars[(common_node.node_id, common_option)])
                model.addConstr(variable >= activation_vars[fixed_node.node_id] + geometry_vars[(common_node.node_id, common_option)] - 1)
                objective_terms.append(
                    -geometry_same_frame_score(
                        left_option=common_option if left_node.is_common_supported else fixed_option,
                        right_option=common_option if right_node.is_common_supported else fixed_option,
                        source_names=source_names,
                        source_weight=config.geometry_source_weight,
                    )
                    * variable
                )

    LOGGER.info(
        "Consensus joint ILP: nodes=%s, common_supported=%s, source_specific=%s, move=%s, division=%s, same_frame_pairs=%s, constraints=%s.",
        len(nodes),
        len(common_node_ids),
        len(source_node_ids),
        len(move_vars),
        len(division_vars),
        len(same_frame_pairs),
        constraint_count,
    )
    model.setObjective(quicksum(objective_terms), GRB.MINIMIZE)
    LOGGER.info("Starting Gurobi optimization for joint consensus/geometry ILP.")
    model.optimize()
    _assert_usable_status(model)

    selected_nodes = {node_id for node_id, variable in activation_vars.items() if variable.X > 0.5}
    incoming_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None] = {}
    outgoing_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None] = {}
    for node_id in selected_nodes:
        incoming_choice[node_id] = None
        outgoing_choice[node_id] = None
    for (parent_id, child_id), variable in move_vars.items():
        if variable.X > 0.5:
            outgoing_choice[parent_id] = ("move", child_id)
            incoming_choice[child_id] = ("move", parent_id)
    for (parent_id, child_id_1, child_id_2), variable in division_vars.items():
        if variable.X > 0.5:
            outgoing_choice[parent_id] = ("division", child_id_1, child_id_2)
            incoming_choice[child_id_1] = ("division", parent_id, child_id_2)
            incoming_choice[child_id_2] = ("division", parent_id, child_id_1)
    for node_id, variable in appearance_vars.items():
        if node_id in selected_nodes and variable.X > 0.5:
            incoming_choice[node_id] = None
    for node_id, variable in disappearance_vars.items():
        if node_id in selected_nodes and variable.X > 0.5:
            outgoing_choice[node_id] = None

    assignments = {
        node_id: option_name
        for (node_id, option_name), variable in geometry_vars.items()
        if node_id in selected_nodes and variable.X > 0.5
    }

    selected_source_counts: dict[str, int] = defaultdict(int)
    selected_common_count = 0
    cross_source_continuations = 0
    for node_id in selected_nodes:
        node = node_lookup[node_id]
        if node.is_common_supported:
            selected_common_count += 1
        elif node.source_name is not None:
            selected_source_counts[node.source_name] += 1
    for (parent_id, child_id), variable in move_vars.items():
        if variable.X <= 0.5:
            continue
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        if dominant_source(parent) != dominant_source(child):
            cross_source_continuations += 1

    selected_graph_stats = {
        "selected_total_count": len(selected_nodes),
        "selected_common_supported_count": selected_common_count,
        "selected_source_specific_count_by_source": dict(sorted(selected_source_counts.items())),
        "selected_cross_source_continuation_count": cross_source_continuations,
    }
    geometry_diag = geometry_diagnostics(
        common_geometry_assignments=assignments,
        node_lookup=node_lookup,
        source_names=source_names,
        indexed_solutions=indexed_solutions,
        selected_nodes=selected_nodes,
        temporal_relations=selected_temporal_relations(selected_nodes, outgoing_choice),
        same_frame_pairs=same_frame_pairs,
        source_weight=config.geometry_source_weight,
        temporal_weight=config.geometry_temporal_overlap_weight,
        neighbor_radius=config.geometry_neighbor_radius,
        cache=geometry_cache,
    )
    LOGGER.info("Consensus joint ILP selected %s fragment node(s) with %s geometry assignment(s).", len(selected_nodes), len(assignments))
    return selected_nodes, incoming_choice, outgoing_choice, selected_graph_stats, assignments, geometry_diag


def geometry_temporal_score(
    parent: TrackletNode,
    parent_option: str,
    child: TrackletNode,
    child_option: str,
    source_names: tuple[str, str],
    indexed_solutions: dict[str, SolutionIndex],
    source_weight: float,
    temporal_weight: float,
    cache: GeometryQueryCache | None = None,
) -> float:
    return (
        source_weight * geometry_source_agreement(parent_option, child_option, source_names)
        + temporal_weight * temporal_boundary_iou(parent, parent_option, child, child_option, indexed_solutions, cache=cache)
    )


def geometry_same_frame_score(
    left_option: str,
    right_option: str,
    source_names: tuple[str, str],
    source_weight: float,
) -> float:
    return source_weight * geometry_source_agreement(left_option, right_option, source_names)


def geometry_diagnostics(
    common_geometry_assignments: dict[int, str],
    node_lookup: dict[int, TrackletNode],
    source_names: tuple[str, str],
    indexed_solutions: dict[str, SolutionIndex],
    selected_nodes: set[int],
    temporal_relations: tuple[tuple[str, int, int, float], ...],
    same_frame_pairs: tuple[tuple[int, int], ...],
    source_weight: float,
    temporal_weight: float,
    neighbor_radius: int,
    cache: GeometryQueryCache | None = None,
) -> dict[str, object]:
    option_counts: dict[str, int] = defaultdict(int)
    source_scores: list[float] = []
    temporal_bonus_values: list[float] = []
    option_names = geometry_option_names(source_names)
    selected_source_ids = tuple(
        sorted(
            node_id
            for node_id in selected_nodes
            if not node_lookup[node_id].is_common_supported
        )
    )
    selected_common_ids = tuple(
        sorted(
            node_id
            for node_id in common_geometry_assignments
            if node_id in selected_nodes and node_lookup[node_id].is_common_supported
        )
    )
    LOGGER.info(
        "Computing geometry diagnostics: selected_common=%s, selected_source=%s, temporal_relations=%s, same_frame_pairs=%s.",
        len(selected_common_ids),
        len(selected_source_ids),
        len(temporal_relations),
        len(same_frame_pairs),
    )

    for option_name in common_geometry_assignments.values():
        option_counts[option_name] += 1

    for _kind, parent_id, child_id, _weight in temporal_relations:
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        parent_option = common_geometry_assignments[parent_id] if parent_id in common_geometry_assignments else fixed_geometry_option(parent)
        child_option = common_geometry_assignments[child_id] if child_id in common_geometry_assignments else fixed_geometry_option(child)
        source_scores.append(geometry_source_agreement(parent_option, child_option, source_names))
        temporal_bonus_values.append(
            temporal_boundary_iou(parent, parent_option, child, child_option, indexed_solutions, cache=cache)
        )

    for left_id, right_id in same_frame_pairs:
        left = node_lookup[left_id]
        right = node_lookup[right_id]
        if left_id not in selected_nodes or right_id not in selected_nodes:
            continue
        left_option = common_geometry_assignments[left_id] if left_id in common_geometry_assignments else fixed_geometry_option(left)
        right_option = common_geometry_assignments[right_id] if right_id in common_geometry_assignments else fixed_geometry_option(right)
        source_scores.append(geometry_source_agreement(left_option, right_option, source_names))

    ruled_out_options = 0
    if selected_common_ids:
        frame_count = len(next(iter(indexed_solutions.values())).solution.frames)
        selected_candidate_nodes = tuple(node_lookup[node_id] for node_id in sorted(selected_nodes))
        common_source_candidates, common_common_candidates = build_common_overlap_candidate_pairs(
            nodes=selected_candidate_nodes,
            frame_count=frame_count,
        )
        common_neighbor_ids: dict[int, list[int]] = defaultdict(list)
        for left_id, right_id in common_common_candidates:
            common_neighbor_ids[left_id].append(right_id)
            common_neighbor_ids[right_id].append(left_id)
        LOGGER.info(
            "Geometry diagnostics candidate pruning: common/source=%s, common/common=%s.",
            sum(len(source_ids) for source_ids in common_source_candidates.values()),
            len(common_common_candidates),
        )

        for node_id in progress_iter(
            selected_common_ids,
            desc="Geometry diagnostics: ruled-out option scan",
            total=len(selected_common_ids),
            unit="common fragment",
        ):
            node = node_lookup[node_id]
            assigned_option = common_geometry_assignments.get(node_id)
            candidate_source_ids = common_source_candidates.get(node_id, ())
            candidate_common_ids = common_neighbor_ids.get(node_id, ())
            for option_name in option_names:
                if option_name == assigned_option:
                    continue
                blocked_by_source = any(
                    option_pair_overlaps_any_shared_frame(
                        node,
                        option_name,
                        node_lookup[source_node_id],
                        fixed_geometry_option(node_lookup[source_node_id]),
                        indexed_solutions,
                        cache=cache,
                    )
                    for source_node_id in candidate_source_ids
                )
                blocked_by_common = any(
                    option_pair_overlaps_any_shared_frame(
                        node,
                        option_name,
                        node_lookup[other_node_id],
                        common_geometry_assignments[other_node_id],
                        indexed_solutions,
                        cache=cache,
                    )
                    for other_node_id in candidate_common_ids
                )
                if blocked_by_source or blocked_by_common:
                    ruled_out_options += 1

    return {
        "assignment_count_by_option": dict(sorted(option_counts.items())),
        "average_source_consistency_score": float(np.mean(source_scores)) if source_scores else 0.0,
        "average_temporal_overlap_bonus": float(np.mean(temporal_bonus_values)) if temporal_bonus_values else 0.0,
        "ruled_out_option_count": int(ruled_out_options),
        "neighbor_pair_count": int(len(same_frame_pairs)),
        "geometry_source_weight": float(source_weight),
        "geometry_temporal_overlap_weight": float(temporal_weight),
        "geometry_neighbor_radius": int(neighbor_radius),
    }


def build_common_overlap_candidate_pairs(
    nodes: tuple[TrackletNode, ...],
    *,
    frame_count: int,
    candidate_node_ids: set[int] | None = None,
) -> tuple[dict[int, tuple[int, ...]], tuple[tuple[int, int], ...]]:
    common_ids_by_frame: list[set[int]] = [set() for _ in range(frame_count)]
    source_ids_by_frame: list[set[int]] = [set() for _ in range(frame_count)]
    for node in nodes:
        if candidate_node_ids is not None and node.node_id not in candidate_node_ids:
            continue
        target = common_ids_by_frame if node.is_common_supported else source_ids_by_frame
        for frame_index in range(node.begin, node.end + 1):
            target[frame_index].add(node.node_id)

    common_source_candidates: dict[int, set[int]] = defaultdict(set)
    common_common_candidates: set[tuple[int, int]] = set()
    for frame_index in range(frame_count):
        common_ids = tuple(sorted(common_ids_by_frame[frame_index]))
        source_ids = source_ids_by_frame[frame_index]
        for common_id in common_ids:
            common_source_candidates[common_id].update(source_ids)
        for left_id, right_id in combinations(common_ids, 2):
            common_common_candidates.add((left_id, right_id))

    return (
        {node_id: tuple(sorted(source_ids)) for node_id, source_ids in common_source_candidates.items() if source_ids},
        tuple(sorted(common_common_candidates)),
    )


def find_same_frame_neighbor_pairs(
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
    source_names: tuple[str, str],
    radius: int,
    selected_node_ids: set[int] | None,
    cache: GeometryQueryCache | None = None,
    progress_desc: str | None = None,
) -> tuple[tuple[int, int], ...]:
    if radius < 0:
        return ()
    frame_count = len(next(iter(indexed_solutions.values())).solution.frames)
    active_nodes_by_frame: list[list[int]] = [[] for _ in range(frame_count)]
    node_lookup = {node.node_id: node for node in nodes}
    for node in nodes:
        if selected_node_ids is not None and node.node_id not in selected_node_ids:
            continue
        for frame_index in range(node.begin, node.end + 1):
            active_nodes_by_frame[frame_index].append(node.node_id)

    candidate_pairs: set[tuple[int, int]] = set()
    for node in nodes:
        if not node.is_common_supported:
            continue
        if selected_node_ids is not None and node.node_id not in selected_node_ids:
            continue
        for frame_index in range(node.begin, node.end + 1):
            for other_node_id in active_nodes_by_frame[frame_index]:
                if other_node_id == node.node_id:
                    continue
                left_id, right_id = sorted((node.node_id, other_node_id))
                pair_key = (left_id, right_id)
                if pair_key in candidate_pairs:
                    continue
                candidate_pairs.add(pair_key)

    sorted_pairs = tuple(sorted(candidate_pairs))
    if progress_desc is not None:
        LOGGER.info("%s: evaluating %s temporal-overlap candidate pair(s).", progress_desc, len(sorted_pairs))
        pair_iterable = progress_iter(sorted_pairs, desc=progress_desc, total=len(sorted_pairs), unit="pair")
    else:
        pair_iterable = sorted_pairs

    confirmed_pairs: list[tuple[int, int]] = []
    for left_id, right_id in pair_iterable:
        if option_pair_within_radius_any_shared_frame(
            node_lookup[left_id],
            node_lookup[right_id],
            indexed_solutions=indexed_solutions,
            source_names=source_names,
            radius=radius,
            cache=cache,
        ):
            confirmed_pairs.append((left_id, right_id))
    return tuple(confirmed_pairs)


def option_pair_within_radius_any_shared_frame(
    left_node: TrackletNode,
    right_node: TrackletNode,
    *,
    indexed_solutions: dict[str, SolutionIndex],
    source_names: tuple[str, str],
    radius: int,
    cache: GeometryQueryCache | None = None,
) -> bool:
    begin = max(left_node.begin, right_node.begin)
    end = min(left_node.end, right_node.end)
    if begin > end:
        return False
    for left_option in node_geometry_options(left_node, source_names):
        for right_option in node_geometry_options(right_node, source_names):
            for frame_index in range(begin, end + 1):
                left_bbox = node_bbox_for_option(left_node, left_option, frame_index, indexed_solutions, cache=cache)
                right_bbox = node_bbox_for_option(right_node, right_option, frame_index, indexed_solutions, cache=cache)
                if not bboxes_within_radius(left_bbox, right_bbox, radius):
                    continue
                left_coords = node_coords_for_option(left_node, left_option, frame_index, indexed_solutions, cache=cache)
                right_coords = node_coords_for_option(right_node, right_option, frame_index, indexed_solutions, cache=cache)
                if coords_within_radius(left_coords, right_coords, radius):
                    return True
    return False


def option_pair_overlaps_any_shared_frame(
    left_node: TrackletNode,
    left_option: str,
    right_node: TrackletNode,
    right_option: str,
    indexed_solutions: dict[str, SolutionIndex],
    cache: GeometryQueryCache | None = None,
) -> bool:
    begin = max(left_node.begin, right_node.begin)
    end = min(left_node.end, right_node.end)
    if begin > end:
        return False
    for frame_index in range(begin, end + 1):
        left_bbox = node_bbox_for_option(left_node, left_option, frame_index, indexed_solutions, cache=cache)
        right_bbox = node_bbox_for_option(right_node, right_option, frame_index, indexed_solutions, cache=cache)
        if not bboxes_overlap(left_bbox, right_bbox):
            continue
        left_coords = node_coord_set_for_option(left_node, left_option, frame_index, indexed_solutions, cache=cache)
        right_coords = node_coord_set_for_option(right_node, right_option, frame_index, indexed_solutions, cache=cache)
        if left_coords and right_coords and coords_overlap(left_coords, right_coords):
            return True
    return False


def temporal_boundary_iou(
    parent: TrackletNode,
    parent_option: str,
    child: TrackletNode,
    child_option: str,
    indexed_solutions: dict[str, SolutionIndex],
    cache: GeometryQueryCache | None = None,
) -> float:
    parent_coords = node_coords_for_option(parent, parent_option, parent.end, indexed_solutions, cache=cache)
    child_coords = node_coords_for_option(child, child_option, child.begin, indexed_solutions, cache=cache)
    return coords_iou(parent_coords, child_coords)


def node_coords_for_option(
    node: TrackletNode,
    option_name: str,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
    cache: GeometryQueryCache | None = None,
) -> np.ndarray:
    cache_key = (node.node_id, option_name, frame_index)
    if cache is not None:
        cached = cache.coords_by_key.get(cache_key)
        if cached is not None:
            return cached
    if node.is_common_supported:
        assert node.source_names is not None and node.source_track_ids is not None
        left_solution = indexed_solutions[node.source_names[0]]
        right_solution = indexed_solutions[node.source_names[1]]
        tracklet = CommonTracklet(
            tracklet_id=node.node_id,
            begin=node.begin,
            end=node.end,
            source_names=node.source_names,
            source_track_ids=node.source_track_ids,
        )
        coords = common_variant_coords(option_name, tracklet, frame_index, left_solution, right_solution)
        if cache is not None:
            cache.coords_by_key[cache_key] = coords
        return coords

    if option_name != fixed_geometry_option(node):
        raise ValueError(f"Source-specific node {node.node_id} only supports option '{fixed_geometry_option(node)}', found '{option_name}'.")
    assert node.source_name is not None and node.source_track_id is not None
    coords = track_coords(indexed_solutions[node.source_name], frame_index, node.source_track_id)
    if cache is not None:
        cache.coords_by_key[cache_key] = coords
    return coords


def node_bbox_for_option(
    node: TrackletNode,
    option_name: str,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
    *,
    cache: GeometryQueryCache | None = None,
) -> tuple[int, int, int, int] | None:
    cache_key = (node.node_id, option_name, frame_index)
    if cache is not None and cache_key in cache.bbox_by_key:
        return cache.bbox_by_key[cache_key]
    coords = node_coords_for_option(node, option_name, frame_index, indexed_solutions, cache=cache)
    bbox = coords_bbox(coords)
    if cache is not None:
        cache.bbox_by_key[cache_key] = bbox
    return bbox


def node_coord_set_for_option(
    node: TrackletNode,
    option_name: str,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
    *,
    cache: GeometryQueryCache | None = None,
) -> frozenset[tuple[int, int]]:
    cache_key = (node.node_id, option_name, frame_index)
    if cache is not None and cache_key in cache.coord_set_by_key:
        return cache.coord_set_by_key[cache_key]
    coords = node_coords_for_option(node, option_name, frame_index, indexed_solutions, cache=cache)
    coord_set = frozenset((int(coord[0]), int(coord[1])) for coord in np.asarray(coords, dtype=np.int32))
    if cache is not None:
        cache.coord_set_by_key[cache_key] = coord_set
    return coord_set


def decode_selected_tracklets(
    nodes: tuple[TrackletNode, ...],
    selected_nodes: set[int],
    incoming_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None],
    outgoing_choice: dict[int, tuple[str, int] | tuple[str, int, int] | None],
) -> tuple[tuple[LineageRecord, ...], dict[int, int]]:
    node_lookup = {node.node_id: node for node in nodes}
    node_to_final_track: dict[int, int] = {}
    lineage_state: dict[int, list[int]] = {}
    next_track_id = 1

    def start_track(node_id: int, parent_track_id: int) -> int:
        nonlocal next_track_id
        node = node_lookup[node_id]
        final_track_id = next_track_id
        next_track_id += 1
        lineage_state[final_track_id] = [node.begin, node.end, parent_track_id]
        node_to_final_track[node_id] = final_track_id
        return final_track_id

    def propagate(node_id: int, final_track_id: int) -> None:
        node = node_lookup[node_id]
        node_to_final_track[node_id] = final_track_id
        lineage_state[final_track_id][1] = max(lineage_state[final_track_id][1], node.end)
        outgoing = outgoing_choice.get(node_id)
        if outgoing is None:
            return
        if outgoing[0] == "move":
            child_id = int(outgoing[1])
            if child_id in node_to_final_track:
                return
            propagate(child_id, final_track_id)
            return

        child_id_1 = int(outgoing[1])
        child_id_2 = int(outgoing[2])
        child_ids = sorted(
            [child_id_1, child_id_2],
            key=lambda child_id: (
                node_lookup[child_id].begin,
                node_lookup[child_id].start_stats.centroid_row,
                node_lookup[child_id].start_stats.centroid_col,
                child_id,
            ),
        )
        for child_id in child_ids:
            child_track_id = start_track(child_id, final_track_id)
            propagate(child_id, child_track_id)

    root_nodes = sorted(
        [node_lookup[node_id] for node_id in selected_nodes if incoming_choice.get(node_id) is None],
        key=lambda node: (
            node.begin,
            node.start_stats.centroid_row,
            node.start_stats.centroid_col,
            node.node_id,
        ),
    )
    for root in root_nodes:
        if root.node_id in node_to_final_track:
            continue
        root_track_id = start_track(root.node_id, 0)
        propagate(root.node_id, root_track_id)

    rows = tuple(
        LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent)
        for track_id, (begin, end, parent) in sorted(lineage_state.items())
    )
    return rows, node_to_final_track


def render_variant_masks(
    variant_name: str,
    raw_frames: np.ndarray,
    nodes: tuple[TrackletNode, ...],
    selected_nodes: set[int],
    node_to_final_track: dict[int, int],
    indexed_solutions: dict[str, SolutionIndex],
    common_geometry_assignments: dict[int, str] | None = None,
) -> np.ndarray:
    tracked_masks = np.zeros((len(raw_frames), *raw_frames[0].shape), dtype=np.uint16)
    selected_node_list = list(node for node in nodes if node.node_id in selected_nodes)
    selected_node_list.sort(
        key=lambda item: (
            node_to_final_track[item.node_id],
            item.begin,
            0 if item.is_common_supported else 1,
            item.node_id,
        )
    )

    for node in selected_node_list:
        track_id = node_to_final_track[node.node_id]
        for frame_index in range(node.begin, node.end + 1):
            if common_geometry_assignments is not None and node.is_common_supported:
                assigned_option = common_geometry_assignments.get(node.node_id)
                if assigned_option is None:
                    raise ValueError(f"Missing geometry assignment for selected common-supported fragment {node.node_id}.")
                coords = node_coords_for_option(node, assigned_option, frame_index, indexed_solutions)
            else:
                coords = node_variant_coords(variant_name, node, frame_index, indexed_solutions)
            if coords.size == 0:
                continue
            frame_mask = tracked_masks[frame_index]
            coords = resolve_render_conflicts(
                variant_name=variant_name,
                node=node,
                track_id=track_id,
                frame_index=frame_index,
                coords=coords,
                frame_mask=frame_mask,
                indexed_solutions=indexed_solutions,
            )
            if coords.size == 0:
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = np.uint16(track_id)
    return tracked_masks


def resolve_render_conflicts(
    variant_name: str,
    node: TrackletNode,
    track_id: int,
    frame_index: int,
    coords: np.ndarray,
    frame_mask: np.ndarray,
    indexed_solutions: dict[str, SolutionIndex],
) -> np.ndarray:
    existing_labels = frame_mask[coords[:, 0], coords[:, 1]]
    available = (existing_labels == 0) | (existing_labels == track_id)
    if np.all(available):
        return coords

    if variant_name == "union" and node.is_common_supported:
        clipped_coords = np.asarray(coords[available], dtype=np.int32)
        clipped_pixels = int(np.count_nonzero(~available))
        if clipped_coords.size > 0:
            LOGGER.warning(
                "Variant '%s' clipped %s pixel(s) from common-supported fragment %s at frame %03d to avoid overlap.",
                variant_name,
                clipped_pixels,
                node.node_id,
                frame_index,
            )
            return clipped_coords

        fallback_coords = node_variant_coords("intersection", node, frame_index, indexed_solutions)
        if fallback_coords.size > 0:
            fallback_existing = frame_mask[fallback_coords[:, 0], fallback_coords[:, 1]]
            fallback_available = (fallback_existing == 0) | (fallback_existing == track_id)
            fallback_coords = np.asarray(fallback_coords[fallback_available], dtype=np.int32)
            if fallback_coords.size > 0:
                LOGGER.warning(
                    "Variant '%s' fell back to intersection geometry for common-supported fragment %s at frame %03d because union geometry was fully occupied.",
                    variant_name,
                    node.node_id,
                    frame_index,
                )
                return fallback_coords

    raise ValueError(
        f"Variant '{variant_name}' contains overlapping selected tracklets at frame {frame_index}."
    )


def summarize_variant(
    variant_name: str,
    tracked_masks: np.ndarray,
    lineage_rows: tuple[LineageRecord, ...],
    common_tracklet_count: int,
    selected_common_tracklets: int,
) -> dict[str, float]:
    object_counts = [int(len(np.unique(frame[frame > 0]))) for frame in tracked_masks]
    return {
        "track_count": float(len(lineage_rows)),
        "division_count": float(sum(1 for row in lineage_rows if row.parent > 0)),
        "frame_count": float(len(tracked_masks)),
        "frame_object_count_min": float(min(object_counts) if object_counts else 0),
        "frame_object_count_mean": float(np.mean(object_counts) if object_counts else 0.0),
        "frame_object_count_max": float(max(object_counts) if object_counts else 0),
        "common_tracklet_coverage": float(selected_common_tracklets / common_tracklet_count) if common_tracklet_count else 1.0,
    }


def input_solution_metrics(
    indexed_solutions: dict[str, SolutionIndex],
    dataset_root: Path,
    track_sequence: str,
) -> dict[str, dict[str, object]]:
    try:
        raw_frames = projectio.load_raw_sequence(dataset_root, track_sequence)
        gt_frames = projectio.load_gt_frame_objects(dataset_root, track_sequence, raw_frames)
        gt_rows = tuple(projectio.load_lineage_records(dataset_root, track_sequence).values())
        gt_solution = build_solution_index(
            SavedTrackingSolution(
                source_name="gt",
                output_dir=dataset_root / f"{track_sequence}_GT" / "TRA",
                tracked_masks=np.stack([frame.label_image.astype(np.uint16) for frame in gt_frames]),
                lineage_rows=gt_rows,
                frames=tuple(gt_frames),
                checkpoint=None,
            )
        )
    except ValueError:
        return {}

    metrics_by_source: dict[str, dict[str, object]] = {}
    source_items = tuple(sorted(indexed_solutions.items()))
    for source_name, solution in progress_iter(
        source_items,
        desc="Evaluating saved input solutions",
        total=len(source_items),
        unit="source",
    ):
        LOGGER.info("Running GT evaluation for saved input solution '%s'.", source_name)
        ctc_payload = evaluate_result_with_ctc(
            dataset_root / f"{track_sequence}_GT",
            solution.solution.output_dir,
        )
        log_ctc_evaluation(f"Consensus input '{source_name}'", ctc_payload)
        metrics_by_source[source_name] = {
            "legacy_gt_metrics": evaluate_solution_against_gt(solution, gt_solution, GT_EVAL_IOU_THRESHOLD),
            "ctc_evaluation": ctc_payload,
        }
    return metrics_by_source


def agreement_metrics(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
    common_tracklets: tuple[CommonTracklet, ...],
) -> dict[str, float]:
    total_objects_1 = sum(frame.object_count for frame in solution_1.solution.frames)
    total_objects_2 = sum(frame.object_count for frame in solution_2.solution.frames)
    matched_vertices = sum(len(frame_matches.pairs) for frame_matches in matches_by_frame)
    shared_moves = sum(max(0, tracklet.end - tracklet.begin) for tracklet in common_tracklets)
    total_moves_1 = sum(max(0, row.end - row.begin) for row in solution_1.solution.lineage_rows)
    total_moves_2 = sum(max(0, row.end - row.begin) for row in solution_2.solution.lineage_rows)
    shared_divisions = count_common_divisions(solution_1, solution_2, matches_by_frame)
    total_divisions_1 = sum(1 for children in solution_1.children_by_parent.values() if len(children) == 2)
    total_divisions_2 = sum(1 for children in solution_2.children_by_parent.values() if len(children) == 2)
    coverage_denominator = max(1, min(total_objects_1, total_objects_2))
    return {
        "object_precision": safe_ratio(matched_vertices, total_objects_1),
        "object_recall": safe_ratio(matched_vertices, total_objects_2),
        "object_f1": f1_score(safe_ratio(matched_vertices, total_objects_1), safe_ratio(matched_vertices, total_objects_2)),
        "move_precision": safe_ratio(shared_moves, total_moves_1),
        "move_recall": safe_ratio(shared_moves, total_moves_2),
        "move_f1": f1_score(safe_ratio(shared_moves, total_moves_1), safe_ratio(shared_moves, total_moves_2)),
        "division_precision": safe_ratio(shared_divisions, total_divisions_1),
        "division_recall": safe_ratio(shared_divisions, total_divisions_2),
        "division_f1": f1_score(safe_ratio(shared_divisions, total_divisions_1), safe_ratio(shared_divisions, total_divisions_2)),
        "shared_tracklet_coverage": safe_ratio(sum(tracklet.end - tracklet.begin + 1 for tracklet in common_tracklets), coverage_denominator),
    }


def evaluate_solution_against_gt(
    solution: SolutionIndex,
    gt_solution: SolutionIndex,
    iou_threshold: float,
) -> dict[str, float]:
    matches_by_frame = match_solutions_by_frame(solution, gt_solution, iou_threshold)
    total_pred_vertices = sum(frame.object_count for frame in solution.solution.frames)
    total_gt_vertices = sum(frame.object_count for frame in gt_solution.solution.frames)
    matched_vertices = sum(len(frame_matches.pairs) for frame_matches in matches_by_frame)

    gt_edges = gt_edge_set(gt_solution)
    predicted_edges = predicted_edge_set(solution, matches_by_frame)
    true_positive_edges = len(predicted_edges & gt_edges)

    ct = complete_tracks(solution, gt_solution, matches_by_frame)
    tf = track_fractions(solution, gt_solution, matches_by_frame)
    bc0 = branching_correctness(solution, gt_solution, matches_by_frame)

    precision = safe_ratio(matched_vertices, total_pred_vertices)
    recall = safe_ratio(matched_vertices, total_gt_vertices)
    link_precision = safe_ratio(true_positive_edges, len(all_solution_edges(solution)))
    link_recall = safe_ratio(true_positive_edges, len(gt_edges))
    return {
        "vertex_precision": precision,
        "vertex_recall": recall,
        "vertex_f1": f1_score(precision, recall),
        "link_precision": link_precision,
        "link_recall": link_recall,
        "link_f1": f1_score(link_precision, link_recall),
        "CT": ct,
        "TF": tf,
        "BC(0)": bc0,
        "BIO": float(np.mean([ct, tf, bc0])),
    }


def candidate_oracle_metrics(
    config: TrackingConfig,
    indexed_solutions: dict[str, SolutionIndex],
    nodes: tuple[TrackletNode, ...],
    continuation_candidates: tuple[ContinuationCandidate, ...],
    division_candidates: tuple[tuple[int, int, int], ...],
) -> dict[str, object]:
    try:
        LOGGER.info("Candidate-oracle diagnostics: loading raw sequence '%s'.", config.track_sequence)
        raw_frames = projectio.load_raw_sequence(config.dataset_root, config.track_sequence)
        LOGGER.info("Candidate-oracle diagnostics: loading GT masks for sequence '%s'.", config.track_sequence)
        gt_frames = projectio.load_gt_frame_objects(config.dataset_root, config.track_sequence, raw_frames)
        LOGGER.info("Candidate-oracle diagnostics: loading GT lineage rows for sequence '%s'.", config.track_sequence)
        gt_rows = tuple(projectio.load_lineage_records(config.dataset_root, config.track_sequence).values())
    except ValueError:
        return {"status": "skipped", "reason": "Ground-truth tracking data is unavailable for candidate-oracle diagnostics."}

    gt_solution = build_solution_index(
        SavedTrackingSolution(
            source_name="gt",
            output_dir=config.dataset_root / f"{config.track_sequence}_GT" / "TRA",
            tracked_masks=np.stack([frame.label_image.astype(np.uint16) for frame in gt_frames]),
            lineage_rows=gt_rows,
            frames=tuple(gt_frames),
            checkpoint=None,
        )
    )

    LOGGER.info(
        "Candidate-oracle diagnostics: matching %s fragment node(s) against %s GT frame(s).",
        len(nodes),
        len(gt_solution.solution.frames),
    )
    gt_matches_by_node = gt_matches_for_nodes(nodes, indexed_solutions, gt_solution, ORACLE_IOU_THRESHOLD)
    gt_vertex_total = sum(frame.object_count for frame in gt_solution.solution.frames)
    covered_vertices = {
        (frame_index, gt_track_id)
        for frame_index, frame_matches in enumerate(gt_matches_by_node)
        for gt_track_id in frame_matches
    }

    covered_moves = 0
    total_moves = 0
    gt_move_requirements = gt_move_requirements_by_frame(gt_solution)
    move_candidate_set = {(candidate.parent_id, candidate.child_id) for candidate in continuation_candidates}
    move_items = tuple(sorted(gt_move_requirements.items()))
    for frame_index, gt_track_ids in progress_iter(
        move_items,
        desc="Checking GT continuation coverage",
        total=len(move_items),
        unit="frame",
    ):
        total_moves += len(gt_track_ids)
        for gt_track_id in gt_track_ids:
            if gt_move_is_covered(frame_index, gt_track_id, gt_matches_by_node, nodes, move_candidate_set):
                covered_moves += 1

    gt_divisions = gt_division_requirements(gt_solution)
    covered_divisions = 0
    for frame_index, parent_id, child_ids in progress_iter(
        gt_divisions,
        desc="Checking GT division coverage",
        total=len(gt_divisions),
        unit="division",
    ):
        if gt_division_is_covered(frame_index, parent_id, child_ids, gt_matches_by_node, division_candidates):
            covered_divisions += 1

    return {
        "status": "success",
        "vertex_coverage": safe_ratio(len(covered_vertices), gt_vertex_total),
        "continuation_edge_coverage": safe_ratio(covered_moves, total_moves),
        "division_edge_coverage": safe_ratio(covered_divisions, len(gt_divisions)),
        "upper_bound_summary": {
            "covered_gt_vertices": len(covered_vertices),
            "total_gt_vertices": gt_vertex_total,
            "covered_gt_continuations": covered_moves,
            "total_gt_continuations": total_moves,
            "covered_gt_divisions": covered_divisions,
            "total_gt_divisions": len(gt_divisions),
        },
    }


def gt_matches_for_nodes(
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
    gt_solution: SolutionIndex,
    iou_threshold: float,
) -> list[dict[int, set[int]]]:
    frame_count = len(gt_solution.solution.frames)
    matches_by_frame: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(frame_count)]
    gt_frames = tuple(gt_solution.solution.frames)
    gt_area_by_frame: list[dict[int, int]] = []
    for _frame_index, gt_frame in progress_iter(
        enumerate(gt_frames),
        desc="Caching GT object areas",
        total=frame_count,
        unit="frame",
    ):
        gt_area_by_frame.append(
            {
                gt_track_id: int(gt_frame.areas[gt_frame.raw_label_to_index[gt_track_id]])
                for gt_track_id in gt_frame.raw_label_ids
            }
        )

    for node in progress_iter(
        nodes,
        desc="Matching fragment nodes to GT",
        total=len(nodes),
        unit="node",
    ):
        for frame_index in range(node.begin, node.end + 1):
            coords_variants = node_oracle_coords(node, frame_index, indexed_solutions)
            if not coords_variants:
                continue
            best_iou_by_gt: dict[int, float] = {}
            gt_frame = gt_frames[frame_index]
            gt_areas = gt_area_by_frame[frame_index]
            for coords in coords_variants:
                for gt_track_id, iou in overlapping_gt_iou_by_label(coords, gt_frame, gt_areas, iou_threshold).items():
                    previous = best_iou_by_gt.get(gt_track_id, 0.0)
                    if iou > previous:
                        best_iou_by_gt[gt_track_id] = iou
            for gt_track_id in best_iou_by_gt:
                matches_by_frame[frame_index][gt_track_id].add(node.node_id)
    return matches_by_frame


def gt_move_requirements_by_frame(gt_solution: SolutionIndex) -> dict[int, set[int]]:
    requirements: dict[int, set[int]] = defaultdict(set)
    for row in gt_solution.solution.lineage_rows:
        for frame_index in range(row.begin, row.end):
            requirements[frame_index].add(row.track_id)
    return requirements


def gt_division_requirements(gt_solution: SolutionIndex) -> list[tuple[int, int, tuple[int, int]]]:
    requirements: list[tuple[int, int, tuple[int, int]]] = []
    for parent_id, children in gt_solution.children_by_parent.items():
        if len(children) != 2:
            continue
        parent_row = gt_solution.rows_by_track[parent_id]
        requirements.append((parent_row.end, parent_id, tuple(sorted(children))))
    return requirements


def gt_move_is_covered(
    frame_index: int,
    gt_track_id: int,
    gt_matches_by_node: list[dict[int, set[int]]],
    nodes: tuple[TrackletNode, ...],
    move_candidate_set: set[tuple[int, int]],
) -> bool:
    node_lookup = {node.node_id: node for node in nodes}
    for node_id in gt_matches_by_node[frame_index].get(gt_track_id, set()):
        node = node_lookup[node_id]
        if node.begin <= frame_index and node.end >= frame_index + 1:
            return True
    for left_id in gt_matches_by_node[frame_index].get(gt_track_id, set()):
        for right_id in gt_matches_by_node[frame_index + 1].get(gt_track_id, set()):
            if (left_id, right_id) in move_candidate_set:
                return True
    return False


def gt_division_is_covered(
    frame_index: int,
    parent_track_id: int,
    child_ids: tuple[int, int],
    gt_matches_by_node: list[dict[int, set[int]]],
    division_candidates: tuple[tuple[int, int, int], ...],
) -> bool:
    parent_matches = gt_matches_by_node[frame_index].get(parent_track_id, set())
    child_matches_1 = gt_matches_by_node[frame_index + 1].get(child_ids[0], set())
    child_matches_2 = gt_matches_by_node[frame_index + 1].get(child_ids[1], set())
    for candidate in division_candidates:
        parent_id, child_id_1, child_id_2 = candidate
        if parent_id not in parent_matches:
            continue
        if {child_id_1, child_id_2} == {next(iter(child_matches_1), None), next(iter(child_matches_2), None)}:
            return True
        if (child_id_1 in child_matches_1 and child_id_2 in child_matches_2) or (child_id_1 in child_matches_2 and child_id_2 in child_matches_1):
            return True
    return False


def node_oracle_coords(
    node: TrackletNode,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
) -> tuple[np.ndarray, ...]:
    if node.is_common_supported:
        return tuple(
            coords
            for variant in ("intersection", "union", node.source_names[0], node.source_names[1])
            if (coords := node_variant_coords(variant, node, frame_index, indexed_solutions)).size > 0
        )
    coords = node_variant_coords("union", node, frame_index, indexed_solutions)
    return (coords,) if coords.size > 0 else ()


def count_common_divisions(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> int:
    common_divisions = 0
    for frame_index, frame_matches in enumerate(matches_by_frame[:-1]):
        next_matches = matches_by_frame[frame_index + 1]
        for pair in frame_matches.pairs:
            children_1 = solution_1.children_by_parent.get(pair.track_id_1, ())
            children_2 = solution_2.children_by_parent.get(pair.track_id_2, ())
            if len(children_1) != 2 or len(children_2) != 2:
                continue
            if any(solution_1.rows_by_track[child].begin != frame_index + 1 for child in children_1):
                continue
            if any(solution_2.rows_by_track[child].begin != frame_index + 1 for child in children_2):
                continue
            mapped_children = {next_matches.source_1_to_2.get(child) for child in children_1}
            if None in mapped_children:
                continue
            if mapped_children == set(children_2):
                common_divisions += 1
    return common_divisions


def gt_edge_set(gt_solution: SolutionIndex) -> set[tuple[str, int, int, int]]:
    edges: set[tuple[str, int, int, int]] = set()
    for row in gt_solution.solution.lineage_rows:
        for frame_index in range(row.begin, row.end):
            edges.add(("move", row.track_id, row.track_id, frame_index))
        if row.parent > 0:
            parent_row = gt_solution.rows_by_track.get(row.parent)
            if parent_row is not None:
                edges.add(("division", row.parent, row.track_id, parent_row.end))
    return edges


def all_solution_edges(solution: SolutionIndex) -> set[tuple[str, int, int, int]]:
    edges: set[tuple[str, int, int, int]] = set()
    for row in solution.solution.lineage_rows:
        for frame_index in range(row.begin, row.end):
            edges.add(("move", row.track_id, row.track_id, frame_index))
        if row.parent > 0:
            parent_row = solution.rows_by_track.get(row.parent)
            if parent_row is not None:
                edges.add(("division", row.parent, row.track_id, parent_row.end))
    return edges


def predicted_edge_set(
    solution: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> set[tuple[str, int, int, int]]:
    gt_edges: set[tuple[str, int, int, int]] = set()
    gt_match_by_frame = [frame_matches.source_1_to_2 for frame_matches in matches_by_frame]
    for row in solution.solution.lineage_rows:
        for frame_index in range(row.begin, row.end):
            gt_track_id = gt_match_by_frame[frame_index].get(row.track_id)
            gt_next_track_id = gt_match_by_frame[frame_index + 1].get(row.track_id)
            if gt_track_id is not None and gt_next_track_id is not None and gt_track_id == gt_next_track_id:
                gt_edges.add(("move", gt_track_id, gt_track_id, frame_index))
        if row.parent > 0:
            parent_row = solution.rows_by_track.get(row.parent)
            if parent_row is None:
                continue
            gt_parent = gt_match_by_frame[parent_row.end].get(row.parent)
            gt_child = gt_match_by_frame[row.begin].get(row.track_id)
            if gt_parent is not None and gt_child is not None:
                gt_edges.add(("division", gt_parent, gt_child, parent_row.end))
    return gt_edges


def complete_tracks(
    solution: SolutionIndex,
    gt_solution: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> float:
    gt_to_pred = [frame_matches.source_2_to_1 for frame_matches in matches_by_frame]
    complete = 0
    for row in gt_solution.solution.lineage_rows:
        mapped = [gt_to_pred[frame_index].get(row.track_id) for frame_index in range(row.begin, row.end + 1)]
        if mapped and None not in mapped and len(set(mapped)) == 1:
            complete += 1
    return safe_ratio(complete, len(gt_solution.solution.lineage_rows))


def track_fractions(
    solution: SolutionIndex,
    gt_solution: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> float:
    gt_to_pred = [frame_matches.source_2_to_1 for frame_matches in matches_by_frame]
    fractions: list[float] = []
    for row in gt_solution.solution.lineage_rows:
        mapped = [gt_to_pred[frame_index].get(row.track_id) for frame_index in range(row.begin, row.end + 1)]
        longest = 0
        current = 0
        last_value: int | None = None
        for value in mapped:
            if value is not None and value == last_value:
                current += 1
            elif value is not None:
                current = 1
                last_value = value
            else:
                current = 0
                last_value = None
            longest = max(longest, current)
        fractions.append(safe_ratio(longest, row.end - row.begin + 1))
    return float(np.mean(fractions)) if fractions else 0.0


def branching_correctness(
    solution: SolutionIndex,
    gt_solution: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> float:
    gt_to_pred = [frame_matches.source_2_to_1 for frame_matches in matches_by_frame]
    correct = 0
    total = 0
    for parent_id, children in gt_solution.children_by_parent.items():
        if len(children) != 2:
            continue
        parent_row = gt_solution.rows_by_track[parent_id]
        frame_index = parent_row.end
        if frame_index + 1 >= len(gt_to_pred):
            continue
        total += 1
        predicted_parent = gt_to_pred[frame_index].get(parent_id)
        predicted_children = {gt_to_pred[frame_index + 1].get(child) for child in children}
        if predicted_parent is None or None in predicted_children:
            continue
        if all(solution.rows_by_track[child].parent == predicted_parent for child in predicted_children):
            correct += 1
    return safe_ratio(correct, total)


def best_input_baseline(input_solution_metrics: dict[str, dict[str, object]]) -> tuple[str | None, float | None]:
    best_source = None
    best_tra = None
    for source_name, payload in input_solution_metrics.items():
        ctc_payload = payload.get("ctc_evaluation", {})
        if not isinstance(ctc_payload, dict):
            continue
        metrics = ctc_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        tra = metrics.get("TRA")
        if not isinstance(tra, (int, float)):
            continue
        if best_tra is None or float(tra) > best_tra:
            best_source = source_name
            best_tra = float(tra)
    return best_source, best_tra


def ctc_metric_deltas(
    variant_ctc_payload: object,
    best_input_tra: float | None,
    best_input_source: str | None,
    input_solution_metrics: dict[str, dict[str, object]],
) -> dict[str, float]:
    if best_input_source is None:
        return {}
    baseline_payload = input_solution_metrics.get(best_input_source, {}).get("ctc_evaluation", {})
    if not isinstance(variant_ctc_payload, dict) or not isinstance(baseline_payload, dict):
        return {}
    variant_metrics = variant_ctc_payload.get("metrics", {})
    baseline_metrics = baseline_payload.get("metrics", {})
    if not isinstance(variant_metrics, dict) or not isinstance(baseline_metrics, dict):
        return {}
    deltas: dict[str, float] = {}
    for key in ("DET", "SEG", "LNK", "TRA", "CT", "TF", "CCA", "AOGM_NS", "AOGM_FN", "AOGM_FP", "AOGM_ED", "AOGM_EA", "AOGM_EC"):
        left = variant_metrics.get(key)
        right = baseline_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = float(left) - float(right)
    if best_input_tra is not None and isinstance(variant_metrics.get("TRA"), (int, float)):
        deltas["delta_to_best_TRA"] = float(variant_metrics["TRA"]) - best_input_tra
    return deltas


def average_match_iou(tracklet: CommonTracklet, matches_by_frame: tuple[FrameMatches, ...]) -> float:
    values = [
        match_iou_at_frame(matches_by_frame[frame_index], tracklet.source_track_ids)
        for frame_index in range(tracklet.begin, tracklet.end + 1)
    ]
    return float(np.mean(values)) if values else 0.0


def match_iou_at_frame(frame_matches: FrameMatches, track_ids: tuple[int, int]) -> float:
    return float(frame_matches.pair_iou.get(track_ids, 0.0))


def fragment_bonus(
    node: TrackletNode,
    activation: gp.Var,
    scorers: EventScorers,
    indexed_solutions: dict[str, SolutionIndex],
    internal_consistency_cache: dict[int, float],
) -> gp.LinExpr:
    internal_consistency = node_internal_consistency(node, scorers, indexed_solutions, internal_consistency_cache)
    if node.is_common_supported:
        confidence = float(np.mean([internal_consistency, node.mean_iou, node.agreement_strength]))
        scale = COMMON_FRAGMENT_BONUS_SCALE
    else:
        confidence = internal_consistency
        scale = SOURCE_FRAGMENT_BONUS_SCALE
    return (-scale * confidence * node.frame_count) * activation


def node_internal_consistency(
    node: TrackletNode,
    scorers: EventScorers,
    indexed_solutions: dict[str, SolutionIndex],
    cache: dict[int, float],
) -> float:
    cached = cache.get(node.node_id)
    if cached is not None:
        return cached
    if node.frame_count <= 1:
        cache[node.node_id] = 0.5
        return 0.5

    probabilities: list[float] = []
    if node.is_common_supported:
        assert node.source_names is not None and node.source_track_ids is not None
        for source_name, track_id in zip(node.source_names, node.source_track_ids, strict=True):
            probabilities.extend(internal_move_probabilities(indexed_solutions[source_name], track_id, node.begin, node.end, scorers))
    else:
        assert node.source_name is not None and node.source_track_id is not None
        probabilities.extend(internal_move_probabilities(indexed_solutions[node.source_name], node.source_track_id, node.begin, node.end, scorers))

    consistency = float(np.mean(probabilities)) if probabilities else 0.5
    cache[node.node_id] = consistency
    return consistency


def internal_move_probabilities(
    solution: SolutionIndex,
    track_id: int,
    begin: int,
    end: int,
    scorers: EventScorers,
) -> list[float]:
    probabilities: list[float] = []
    for frame_index in range(begin, end):
        left = object_stats(solution, frame_index, track_id)
        right = object_stats(solution, frame_index + 1, track_id)
        features = (
            left.intensity_std,
            right.intensity_std,
            centroid_distance(left, right),
            left.area,
            right.area,
        )
        probability = positive_probability(scorers.move_model, features)
        probabilities.append(probability)
    return probabilities


def positive_probability(model: object, features: tuple[float, ...]) -> float:
    probability = float(model.predict_proba(np.asarray([features], dtype=float))[0, 1])
    return max(0.0, min(1.0, probability))


def move_cost(scorers: EventScorers, parent: TrackletNode, child: TrackletNode) -> float:
    features = (
        parent.end_stats.intensity_std,
        child.start_stats.intensity_std,
        centroid_distance(parent.end_stats, child.start_stats),
        parent.end_stats.area,
        child.start_stats.area,
    )
    return probability_to_cost(scorers.move_model, features)


def division_cost(scorers: EventScorers, parent: TrackletNode, child_1: TrackletNode, child_2: TrackletNode) -> float:
    features = (
        parent.end_stats.intensity_std,
        child_1.start_stats.intensity_std,
        child_2.start_stats.intensity_std,
        centroid_distance(parent.end_stats, child_1.start_stats),
        centroid_distance(parent.end_stats, child_2.start_stats),
        centroid_distance(child_1.start_stats, child_2.start_stats),
        parent.end_stats.area,
        child_1.start_stats.area,
        child_2.start_stats.area,
    )
    return probability_to_cost(scorers.division_model, features)


def appearance_cost(scorers: EventScorers, node: TrackletNode) -> float:
    return probability_to_cost(
        scorers.appearance_model,
        (
            node.start_stats.intensity_std,
            node.start_stats.border_distance,
            node.start_stats.area,
        ),
    )


def disappearance_cost(scorers: EventScorers, node: TrackletNode) -> float:
    return probability_to_cost(
        scorers.disappearance_model,
        (
            node.end_stats.intensity_std,
            node.end_stats.border_distance,
            node.end_stats.area,
        ),
    )


def dominant_source(node: TrackletNode) -> str:
    if node.source_name is not None:
        return node.source_name
    assert node.source_names is not None
    return "+".join(node.source_names)


def boundary_overlap_details(
    parent: TrackletNode,
    child: TrackletNode,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
) -> tuple[int, float]:
    parent_coords = node_variant_coords("union", parent, frame_index, indexed_solutions)
    child_coords = node_variant_coords("union", child, frame_index, indexed_solutions)
    if parent_coords.size == 0 or child_coords.size == 0:
        return 0, 0.0
    overlap = intersect_coords(parent_coords, child_coords)
    if overlap.size == 0:
        return 0, 0.0
    smaller = min(len(parent_coords), len(child_coords))
    return int(len(overlap)), float(len(overlap) / max(1, smaller))


def node_variant_coords(
    variant_name: str,
    node: TrackletNode,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
) -> np.ndarray:
    if frame_index < node.begin or frame_index > node.end:
        return np.empty((0, 2), dtype=np.int32)
    if node.is_common_supported:
        assert node.source_names is not None
        assert node.source_track_ids is not None
        left_solution = indexed_solutions[node.source_names[0]]
        right_solution = indexed_solutions[node.source_names[1]]
        tracklet = CommonTracklet(
            tracklet_id=node.node_id,
            begin=node.begin,
            end=node.end,
            source_names=node.source_names,
            source_track_ids=node.source_track_ids,
        )
        return common_variant_coords(variant_name, tracklet, frame_index, left_solution, right_solution)

    assert node.source_name is not None
    assert node.source_track_id is not None
    return track_coords(indexed_solutions[node.source_name], frame_index, node.source_track_id)


def common_variant_coords(
    variant_name: str,
    tracklet: CommonTracklet,
    frame_index: int,
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
) -> np.ndarray:
    coords_1 = track_coords(solution_1, frame_index, tracklet.source_track_ids[0])
    coords_2 = track_coords(solution_2, frame_index, tracklet.source_track_ids[1])
    if variant_name == "intersection":
        return intersect_coords(coords_1, coords_2)
    if variant_name == "union":
        return union_coords(coords_1, coords_2)
    if variant_name == tracklet.source_names[0]:
        return coords_1
    if variant_name == tracklet.source_names[1]:
        return coords_2
    raise ValueError(f"Unknown common-tracklet variant '{variant_name}'.")


def track_coords(solution: SolutionIndex, frame_index: int, track_id: int) -> np.ndarray:
    frame = solution.solution.frames[frame_index]
    object_index = frame.raw_label_to_index.get(track_id)
    if object_index is None:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(frame.coords[object_index], dtype=np.int32)


def overlapping_gt_iou_by_label(
    coords: np.ndarray,
    gt_frame,
    gt_areas: dict[int, int],
    iou_threshold: float,
) -> dict[int, float]:
    if coords.size == 0:
        return {}
    labels = gt_frame.label_image[coords[:, 0], coords[:, 1]]
    labels = labels[labels > 0]
    if labels.size == 0:
        return {}

    unique_labels, intersections = np.unique(labels, return_counts=True)
    coords_area = int(len(coords))
    matches: dict[int, float] = {}
    for gt_track_id_raw, intersection_raw in zip(unique_labels, intersections, strict=True):
        gt_track_id = int(gt_track_id_raw)
        gt_area = int(gt_areas.get(gt_track_id, 0))
        intersection = int(intersection_raw)
        union = coords_area + gt_area - intersection
        if union <= 0:
            continue
        iou = float(intersection / union)
        if iou >= iou_threshold:
            matches[gt_track_id] = iou
    return matches


def object_stats(solution: SolutionIndex, frame_index: int, track_id: int) -> ObjectStats:
    frame = solution.solution.frames[frame_index]
    object_index = frame.raw_label_to_index.get(track_id)
    if object_index is None:
        raise ValueError(f"Track {track_id} is missing from frame {frame_index} in solution '{solution.solution.source_name}'.")
    centroid_row, centroid_col = frame.centroids[object_index]
    return ObjectStats(
        centroid_row=centroid_row,
        centroid_col=centroid_col,
        area=float(frame.areas[object_index]),
        intensity_std=float(frame.intensity_std[object_index]),
        border_distance=float(frame.border_distance[object_index]),
    )


def average_stats(left: ObjectStats, right: ObjectStats) -> ObjectStats:
    return ObjectStats(
        centroid_row=(left.centroid_row + right.centroid_row) / 2.0,
        centroid_col=(left.centroid_col + right.centroid_col) / 2.0,
        area=(left.area + right.area) / 2.0,
        intensity_std=(left.intensity_std + right.intensity_std) / 2.0,
        border_distance=(left.border_distance + right.border_distance) / 2.0,
    )


def centroid_distance(left: ObjectStats, right: ObjectStats) -> float:
    return float(np.hypot(left.centroid_row - right.centroid_row, left.centroid_col - right.centroid_col))


def intersect_coords(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    left_set = {tuple(coord) for coord in np.asarray(left, dtype=np.int32)}
    intersection = sorted(left_set.intersection(tuple(coord) for coord in np.asarray(right, dtype=np.int32)))
    return np.asarray(intersection, dtype=np.int32) if intersection else np.empty((0, 2), dtype=np.int32)


def union_coords(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0:
        return np.asarray(right, dtype=np.int32)
    if right.size == 0:
        return np.asarray(left, dtype=np.int32)
    coords = sorted({tuple(coord) for coord in np.asarray(left, dtype=np.int32)} | {tuple(coord) for coord in np.asarray(right, dtype=np.int32)})
    return np.asarray(coords, dtype=np.int32)


def coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.size == 0:
        return None
    coords = np.asarray(coords, dtype=np.int32)
    coord_min = np.min(coords, axis=0)
    coord_max = np.max(coords, axis=0)
    return (int(coord_min[0]), int(coord_min[1]), int(coord_max[0]), int(coord_max[1]))


def bboxes_overlap(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> bool:
    if left is None or right is None:
        return False
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def bboxes_within_radius(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
    radius: int,
) -> bool:
    if radius < 0 or left is None or right is None:
        return False
    row_gap = max(0, left[0] - right[2], right[0] - left[2])
    col_gap = max(0, left[1] - right[3], right[1] - left[3])
    return row_gap * row_gap + col_gap * col_gap <= radius * radius


def coords_overlap(
    left: frozenset[tuple[int, int]] | set[tuple[int, int]],
    right: frozenset[tuple[int, int]] | set[tuple[int, int]],
) -> bool:
    if not left or not right:
        return False
    if len(left) > len(right):
        left, right = right, left
    return any(coord in right for coord in left)


def coords_within_radius(left: np.ndarray, right: np.ndarray, radius: int) -> bool:
    if radius < 0 or left.size == 0 or right.size == 0:
        return False
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    left_min = np.min(left, axis=0)
    left_max = np.max(left, axis=0)
    right_min = np.min(right, axis=0)
    right_max = np.max(right, axis=0)
    row_gap = max(0, left_min[0] - right_max[0], right_min[0] - left_max[0])
    col_gap = max(0, left_min[1] - right_max[1], right_min[1] - left_max[1])
    if row_gap * row_gap + col_gap * col_gap > radius * radius:
        return False
    row_delta = left[:, None, 0].astype(np.int64) - right[None, :, 0].astype(np.int64)
    col_delta = left[:, None, 1].astype(np.int64) - right[None, :, 1].astype(np.int64)
    return bool(np.any((row_delta * row_delta + col_delta * col_delta) <= radius * radius))


def coords_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 1.0
    if left.size == 0 or right.size == 0:
        return 0.0
    left_set = {tuple(coord) for coord in np.asarray(left, dtype=np.int32)}
    right_set = {tuple(coord) for coord in np.asarray(right, dtype=np.int32)}
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return 0.0 if union == 0 else float(intersection / union)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
