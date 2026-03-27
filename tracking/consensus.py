from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import logging
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum
from scipy.optimize import linear_sum_assignment

from dataio import projectio
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

    @property
    def frame_count(self) -> int:
        return self.end - self.begin + 1


@dataclass(slots=True, frozen=True)
class FrameMatches:
    pairs: tuple[MatchedObjectPair, ...]
    source_1_to_2: dict[int, int]
    source_2_to_1: dict[int, int]


@dataclass(slots=True, frozen=True)
class ConsensusPreparation:
    common_tracklets: tuple[CommonTracklet, ...]
    hypothesis_tracklets: tuple[HypothesisTracklet, ...]
    nodes: tuple[TrackletNode, ...]
    matches_by_frame: tuple[FrameMatches, ...]
    input_metrics: ConsensusMetrics


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
    preparation = prepare_consensus(config, indexed_solutions)
    output_root = projectio.resolve_consensus_output_dir(config)
    output_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Consensus preparation produced %s fixed common tracklet(s) and %s hypothesis tracklet(s).",
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
    }
    write_json(premerge_json_path, premerge_payload)
    write_text(premerge_text_path, format_metrics_report(premerge_payload))
    LOGGER.info("Wrote pre-merge metrics to %s and %s.", premerge_json_path, premerge_text_path)

    selected_nodes, incoming_choice, outgoing_choice = solve_global_tracklet_ilp(
        config=config,
        nodes=preparation.nodes,
        indexed_solutions=indexed_solutions,
        scorers=scorers,
    )
    lineage_rows, node_to_final_track = decode_selected_tracklets(
        nodes=preparation.nodes,
        selected_nodes=selected_nodes,
        incoming_choice=incoming_choice,
        outgoing_choice=outgoing_choice,
    )

    gt_index: SolutionIndex | None = None
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
        gt_index = build_solution_index(gt_solution)
        LOGGER.info("Loaded GT tracking data for consensus evaluation.")
    except ValueError:
        LOGGER.info("No GT tracking data found for %s; consensus evaluation will use summary metrics only.", config.track_sequence)

    variant_names = ("intersection", "union", source_names[0], source_names[1])
    variant_evaluations: dict[str, VariantEvaluation] = {}
    comparison_payload: dict[str, dict[str, float]] = {}
    selected_common_nodes = {
        node_id
        for node_id in selected_nodes
        if preparation.nodes[node_id].fixed
    }
    for variant_name in variant_names:
        LOGGER.info("Rendering consensus variant '%s'.", variant_name)
        tracked_masks = render_variant_masks(
            variant_name=variant_name,
            raw_frames=raw_frames,
            nodes=preparation.nodes,
            selected_nodes=selected_nodes,
            node_to_final_track=node_to_final_track,
            indexed_solutions=indexed_solutions,
        )
        variant_dir = output_root / variant_name
        mask_paths, lineage_path = projectio.write_tracking_outputs(variant_dir, tracked_masks, lineage_rows)
        metrics = summarize_variant(
            variant_name=variant_name,
            tracked_masks=tracked_masks,
            lineage_rows=lineage_rows,
            common_tracklet_count=len(preparation.common_tracklets),
            selected_common_tracklets=len(selected_common_nodes),
        )
        if gt_index is not None:
            metrics.update(
                evaluate_solution_against_gt(
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
            )
        metrics_json_path = variant_dir / "metrics.json"
        metrics_text_path = variant_dir / "metrics.txt"
        write_json(metrics_json_path, metrics)
        write_text(metrics_text_path, format_metrics_report(metrics))
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
        LOGGER.info("Saved consensus variant '%s' to %s.", variant_name, variant_dir)

    comparison_json_path = output_root / "variant_comparison.json"
    comparison_text_path = output_root / "variant_comparison.txt"
    write_json(comparison_json_path, comparison_payload)
    write_text(comparison_text_path, format_metrics_report(comparison_payload))
    LOGGER.info("Wrote variant comparison metrics to %s and %s.", comparison_json_path, comparison_text_path)

    return ConsensusResult(
        selected_sources=source_names,
        lineage_rows=lineage_rows,
        output_dir=output_root,
        premerge_metrics_path=premerge_json_path,
        premerge_metrics_text_path=premerge_text_path,
        variant_comparison_path=comparison_json_path,
        variant_comparison_text_path=comparison_text_path,
        variant_evaluations=variant_evaluations,
    )


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
) -> ConsensusPreparation:
    source_name_1, source_name_2 = config.consensus_sources
    solution_1 = indexed_solutions[source_name_1]
    solution_2 = indexed_solutions[source_name_2]
    matches_by_frame = match_solutions_by_frame(solution_1, solution_2, config.agreement_iou_threshold)
    common_tracklets = build_common_tracklets(solution_1, solution_2, matches_by_frame)
    common_tracklets = tuple(
        tracklet
        for tracklet in common_tracklets
        if common_tracklet_has_non_empty_intersection(tracklet, solution_1, solution_2)
    )
    hypothesis_tracklets = build_hypothesis_tracklets(indexed_solutions, common_tracklets, config.consensus_sources)

    common_nodes: list[TrackletNode] = []
    hypothesis_nodes: list[TrackletNode] = []
    next_node_id = 0
    for tracklet in common_tracklets:
        common_nodes.append(
            TrackletNode(
                node_id=next_node_id,
                begin=tracklet.begin,
                end=tracklet.end,
                fixed=True,
                kind="common",
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
            )
        )
        next_node_id += 1

    for tracklet in hypothesis_tracklets:
        solution = indexed_solutions[tracklet.source_name]
        hypothesis_nodes.append(
            TrackletNode(
                node_id=next_node_id,
                begin=tracklet.begin,
                end=tracklet.end,
                fixed=False,
                kind="hypothesis",
                source_name=tracklet.source_name,
                source_track_id=tracklet.source_track_id,
                source_names=None,
                source_track_ids=None,
                start_stats=object_stats(solution, tracklet.begin, tracklet.source_track_id),
                end_stats=object_stats(solution, tracklet.end, tracklet.source_track_id),
            )
        )
        next_node_id += 1

    kept_hypothesis_nodes = filter_conflicting_hypotheses(common_nodes, hypothesis_nodes, indexed_solutions)
    nodes = tuple(common_nodes + list(kept_hypothesis_nodes))
    kept_signatures = {
        (node.source_name, node.source_track_id, node.begin, node.end)
        for node in kept_hypothesis_nodes
    }
    kept_hypothesis_tracklets = tuple(
        tracklet
        for tracklet in hypothesis_tracklets
        if (tracklet.source_name, tracklet.source_track_id, tracklet.begin, tracklet.end) in kept_signatures
    )
    input_metrics = ConsensusMetrics(
        agreement_metrics=agreement_metrics(solution_1, solution_2, matches_by_frame, common_tracklets),
        input_solution_metrics=input_solution_metrics(indexed_solutions, config.dataset_root, config.track_sequence),
        common_tracklet_count=len(common_tracklets),
        hypothesis_tracklet_count=len(nodes) - len(common_nodes),
    )
    return ConsensusPreparation(
        common_tracklets=common_tracklets,
        hypothesis_tracklets=kept_hypothesis_tracklets,
        nodes=nodes,
        matches_by_frame=matches_by_frame,
        input_metrics=input_metrics,
    )


def match_solutions_by_frame(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    iou_threshold: float,
) -> tuple[FrameMatches, ...]:
    matches: list[FrameMatches] = []
    for frame_1, frame_2 in zip(solution_1.solution.frames, solution_2.solution.frames, strict=True):
        if frame_1.object_count == 0 or frame_2.object_count == 0:
            matches.append(FrameMatches(pairs=(), source_1_to_2={}, source_2_to_1={}))
            continue

        intersections = intersection_areas(frame_1, frame_2)
        cost_matrix = np.ones((frame_1.object_count, frame_2.object_count), dtype=float)
        for (left_track_id, right_track_id), intersection in intersections.items():
            left_index = frame_1.raw_label_to_index[left_track_id]
            right_index = frame_2.raw_label_to_index[right_track_id]
            union = frame_1.areas[left_index] + frame_2.areas[right_index] - intersection
            iou = 0.0 if union == 0 else float(intersection / union)
            cost_matrix[left_index, right_index] = 1.0 - iou

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
            )
        )
    return tuple(matches)


def build_common_tracklets(
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
    matches_by_frame: tuple[FrameMatches, ...],
) -> tuple[CommonTracklet, ...]:
    visited: set[tuple[int, int]] = set()
    tracklets: list[CommonTracklet] = []
    next_tracklet_id = 0
    for frame_index, matches in enumerate(matches_by_frame):
        for pair in matches.pairs:
            visit_key = (frame_index, pair.track_id_1)
            if visit_key in visited:
                continue
            end_frame = frame_index
            track_id_1 = pair.track_id_1
            track_id_2 = pair.track_id_2
            while (
                end_frame + 1 < len(matches_by_frame)
                and end_frame < solution_1.rows_by_track[track_id_1].end
                and end_frame < solution_2.rows_by_track[track_id_2].end
                and matches_by_frame[end_frame + 1].source_1_to_2.get(track_id_1) == track_id_2
            ):
                end_frame += 1
            for common_frame in range(frame_index, end_frame + 1):
                visited.add((common_frame, track_id_1))
            tracklets.append(
                CommonTracklet(
                    tracklet_id=next_tracklet_id,
                    begin=frame_index,
                    end=end_frame,
                    source_names=(solution_1.solution.source_name, solution_2.solution.source_name),
                    source_track_ids=(track_id_1, track_id_2),
                )
            )
            next_tracklet_id += 1
    return tuple(tracklets)


def common_tracklet_has_non_empty_intersection(
    tracklet: CommonTracklet,
    solution_1: SolutionIndex,
    solution_2: SolutionIndex,
) -> bool:
    for frame_index in range(tracklet.begin, tracklet.end + 1):
        coords = common_variant_coords("intersection", tracklet, frame_index, solution_1, solution_2)
        if coords.size == 0:
            return False
    return True


def build_hypothesis_tracklets(
    indexed_solutions: dict[str, SolutionIndex],
    common_tracklets: tuple[CommonTracklet, ...],
    source_names: tuple[str, ...],
) -> tuple[HypothesisTracklet, ...]:
    common_intervals_by_source_track: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for tracklet in common_tracklets:
        common_intervals_by_source_track[(tracklet.source_names[0], tracklet.source_track_ids[0])].append((tracklet.begin, tracklet.end))
        common_intervals_by_source_track[(tracklet.source_names[1], tracklet.source_track_ids[1])].append((tracklet.begin, tracklet.end))

    hypothesis_tracklets: list[HypothesisTracklet] = []
    next_tracklet_id = len(common_tracklets)
    for source_name in source_names:
        solution = indexed_solutions[source_name]
        for row in sorted(solution.solution.lineage_rows, key=lambda record: (record.begin, record.track_id)):
            intervals = sorted(common_intervals_by_source_track.get((source_name, row.track_id), []))
            start = row.begin
            for begin, end in intervals:
                if start < begin:
                    hypothesis_tracklets.append(
                        HypothesisTracklet(
                            tracklet_id=next_tracklet_id,
                            begin=start,
                            end=begin - 1,
                            source_name=source_name,
                            source_track_id=row.track_id,
                        )
                    )
                    next_tracklet_id += 1
                start = max(start, end + 1)
            if start <= row.end:
                hypothesis_tracklets.append(
                    HypothesisTracklet(
                        tracklet_id=next_tracklet_id,
                        begin=start,
                        end=row.end,
                        source_name=source_name,
                        source_track_id=row.track_id,
                    )
                )
                next_tracklet_id += 1
    return tuple(hypothesis_tracklets)


def filter_conflicting_hypotheses(
    common_nodes: list[TrackletNode],
    hypothesis_nodes: list[TrackletNode],
    indexed_solutions: dict[str, SolutionIndex],
) -> tuple[TrackletNode, ...]:
    kept: list[TrackletNode] = []
    for node in hypothesis_nodes:
        if any(nodes_overlap(node, common_node, indexed_solutions) for common_node in common_nodes):
            LOGGER.info(
                "Dropping hypothesis tracklet %s (%s track %s, %03d-%03d) because it conflicts with a fixed common tracklet.",
                node.node_id,
                node.source_name,
                node.source_track_id,
                node.begin,
                node.end,
            )
            continue
        kept.append(node)
    return tuple(kept)


def solve_global_tracklet_ilp(
    config: TrackingConfig,
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
    scorers: EventScorers,
) -> tuple[set[int], dict[int, tuple[str, int] | tuple[str, int, int] | None], dict[int, tuple[str, int] | tuple[str, int, int] | None]]:
    if not nodes:
        return set(), {}, {}

    node_lookup = {node.node_id: node for node in nodes}
    fixed_node_ids = {node.node_id for node in nodes if node.fixed}
    hypothesis_node_ids = {node.node_id for node in nodes if not node.fixed}
    nodes_by_begin: dict[int, list[TrackletNode]] = defaultdict(list)
    conflict_pairs = compute_hypothesis_conflicts(nodes, indexed_solutions)
    for node in nodes:
        nodes_by_begin[node.begin].append(node)

    move_candidates: list[tuple[int, int]] = []
    division_candidates: list[tuple[int, int, int]] = []
    candidate_children_by_parent: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        for child in nodes_by_begin.get(node.end + 1, []):
            if node.node_id == child.node_id:
                continue
            if centroid_distance(node.end_stats, child.start_stats) <= config.max_distance:
                move_candidates.append((node.node_id, child.node_id))
                candidate_children_by_parent[node.node_id].append(child.node_id)
        for child_1, child_2 in combinations(sorted(candidate_children_by_parent[node.node_id]), 2):
            if frozenset((child_1, child_2)) in conflict_pairs:
                continue
            child_node_1 = node_lookup[child_1]
            child_node_2 = node_lookup[child_2]
            if centroid_distance(node.end_stats, child_node_1.start_stats) <= config.max_distance and centroid_distance(node.end_stats, child_node_2.start_stats) <= config.max_distance:
                division_candidates.append((node.node_id, child_1, child_2))

    model = gp.Model("PyTr2dConsensus")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    activation_vars: dict[int, gp.Var] = {}
    appearance_vars: dict[int, gp.Var] = {}
    disappearance_vars: dict[int, gp.Var] = {}
    move_vars: dict[tuple[int, int], gp.Var] = {}
    division_vars: dict[tuple[int, int, int], gp.Var] = {}
    incoming_terms: dict[int, list[gp.Var]] = defaultdict(list)
    outgoing_terms: dict[int, list[gp.Var]] = defaultdict(list)
    objective_terms: list[gp.LinExpr] = []

    for node_id in sorted(hypothesis_node_ids):
        node = node_lookup[node_id]
        activation = model.addVar(vtype=GRB.BINARY, name=f"act[{node_id}]")
        activation_vars[node_id] = activation
        objective_terms.append(config.segmentation_reward * node.frame_count * activation)

    for node in nodes:
        appearance = model.addVar(vtype=GRB.BINARY, name=f"app[{node.node_id}]")
        disappearance = model.addVar(vtype=GRB.BINARY, name=f"dis[{node.node_id}]")
        appearance_vars[node.node_id] = appearance
        disappearance_vars[node.node_id] = disappearance
        objective_terms.append(appearance_cost(scorers, node) * appearance)
        objective_terms.append(disappearance_cost(scorers, node) * disappearance)

    for parent_id, child_id in move_candidates:
        parent = node_lookup[parent_id]
        child = node_lookup[child_id]
        variable = model.addVar(vtype=GRB.BINARY, name=f"move[{parent_id},{child_id}]")
        move_vars[(parent_id, child_id)] = variable
        outgoing_terms[parent_id].append(variable)
        incoming_terms[child_id].append(variable)
        objective_terms.append(move_cost(scorers, parent, child) * variable)

    for parent_id, child_id_1, child_id_2 in division_candidates:
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
    for node in nodes:
        target = 1 if node.fixed else activation_vars[node.node_id]
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

    for left_id, right_id in sorted(conflict_pairs):
        if left_id in fixed_node_ids or right_id in fixed_node_ids:
            continue
        model.addConstr(
            activation_vars[left_id] + activation_vars[right_id] <= 1,
            name=f"conflict[{left_id},{right_id}]",
        )
        constraint_count += 1

    LOGGER.info(
        "Consensus ILP: fixed=%s, hypothesis=%s, move=%s, division=%s, constraints=%s, conflicts=%s.",
        len(fixed_node_ids),
        len(hypothesis_node_ids),
        len(move_vars),
        len(division_vars),
        constraint_count,
        len(conflict_pairs),
    )
    model.setObjective(quicksum(objective_terms), GRB.MINIMIZE)
    LOGGER.info("Starting Gurobi optimization for consensus ILP.")
    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi failed to find a usable consensus solution. Status code: {model.Status}")

    selected_nodes = set(fixed_node_ids)
    for node_id, variable in activation_vars.items():
        if variable.X > 0.5:
            selected_nodes.add(node_id)

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

    LOGGER.info("Consensus ILP selected %s tracklet node(s).", len(selected_nodes))
    return selected_nodes, incoming_choice, outgoing_choice


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
        lineage_state[final_track_id][1] = node.end
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
) -> np.ndarray:
    tracked_masks = np.zeros((len(raw_frames), *raw_frames[0].shape), dtype=np.uint16)
    for node in sorted(
        (node for node in nodes if node.node_id in selected_nodes),
        key=lambda item: (node_to_final_track[item.node_id], item.begin, item.node_id),
    ):
        track_id = node_to_final_track[node.node_id]
        for frame_index in range(node.begin, node.end + 1):
            coords = node_variant_coords(variant_name, node, frame_index, indexed_solutions)
            if coords.size == 0:
                continue
            frame_mask = tracked_masks[frame_index]
            existing_labels = frame_mask[coords[:, 0], coords[:, 1]]
            conflicting = existing_labels[(existing_labels != 0) & (existing_labels != track_id)]
            if conflicting.size > 0:
                raise ValueError(
                    f"Variant '{variant_name}' contains overlapping selected tracklets at frame {frame_index}."
                )
            frame_mask[coords[:, 0], coords[:, 1]] = np.uint16(track_id)
    return tracked_masks


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
) -> dict[str, dict[str, float]]:
    raw_frames = None
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

    return {
        source_name: evaluate_solution_against_gt(solution, gt_solution, GT_EVAL_IOU_THRESHOLD)
        for source_name, solution in indexed_solutions.items()
    }


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


def compute_hypothesis_conflicts(
    nodes: tuple[TrackletNode, ...],
    indexed_solutions: dict[str, SolutionIndex],
) -> set[frozenset[int]]:
    conflicts: set[frozenset[int]] = set()
    for left_node, right_node in combinations(nodes, 2):
        if left_node.fixed or right_node.fixed:
            continue
        if nodes_overlap(left_node, right_node, indexed_solutions):
            conflicts.add(frozenset((left_node.node_id, right_node.node_id)))
    return conflicts


def nodes_overlap(
    left_node: TrackletNode,
    right_node: TrackletNode,
    indexed_solutions: dict[str, SolutionIndex],
) -> bool:
    overlap_begin = max(left_node.begin, right_node.begin)
    overlap_end = min(left_node.end, right_node.end)
    if overlap_begin > overlap_end:
        return False
    for frame_index in range(overlap_begin, overlap_end + 1):
        left_coords = node_variant_coords("union", left_node, frame_index, indexed_solutions)
        right_coords = node_variant_coords("union", right_node, frame_index, indexed_solutions)
        if coords_overlap(left_coords, right_coords):
            return True
    return False


def node_variant_coords(
    variant_name: str,
    node: TrackletNode,
    frame_index: int,
    indexed_solutions: dict[str, SolutionIndex],
) -> np.ndarray:
    if frame_index < node.begin or frame_index > node.end:
        return np.empty((0, 2), dtype=np.int32)
    if node.fixed:
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


def coords_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    if left.size == 0 or right.size == 0:
        return False
    left_set = {tuple(coord) for coord in np.asarray(left, dtype=np.int32)}
    return any(tuple(coord) in left_set for coord in np.asarray(right, dtype=np.int32))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_metrics_report(payload: object, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(format_metrics_report(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    return f"{prefix}{payload}"
