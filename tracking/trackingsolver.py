from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import logging
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from dataio import projectio
from tracking.random_forest import EventScorers, candidate_neighborhoods
from tracking.types import FrameObjects, LineageRecord, TrackingCheckpoint, TrackingConfig, TrackingResult


LOGGER = logging.getLogger(__name__)


CurrentKey = int
NextKey = tuple[str, int]
MoveKey = tuple[int, str, int]
DivisionKey = tuple[int, str, int, int]
CHECKPOINT_VERSION = 1


@dataclass(slots=True)
class ResumeState:
    completed_frame: int
    canonical_frames: list[FrameObjects]
    tracked_masks: np.ndarray
    lineage_state: dict[int, list[int]]
    next_track_id: int


def solve_tracking(
    config: TrackingConfig,
    raw_frames: np.ndarray,
    frames_by_source: dict[str, list[FrameObjects]],
    scorers: EventScorers,
) -> TrackingResult:
    if not frames_by_source:
        raise ValueError("At least one segmentation source is required for tracking.")

    source_names = tuple(frames_by_source.keys())
    frame_count = len(raw_frames)
    reference_shape = tuple(int(value) for value in raw_frames[0].shape)
    _validate_frames(raw_frames, frames_by_source, frame_count, reference_shape)
    output_dir = projectio.resolve_output_dir(config)
    LOGGER.info(
        "Preparing pairwise tracking for %s source(s), %s frame(s), frame shape %s.",
        len(source_names),
        frame_count,
        reference_shape,
    )

    resume_state = maybe_resume_tracking(
        config=config,
        raw_frames=raw_frames,
        output_dir=output_dir,
        source_names=source_names,
        frame_count=frame_count,
        reference_shape=reference_shape,
    )

    if resume_state is None:
        canonical_frames: list[FrameObjects] = []
        tracked_masks = np.zeros((frame_count, *reference_shape), dtype=np.uint16)
        current_frame = initialize_first_frame(
            config,
            raw_frames[0],
            {name: frames[0] for name, frames in frames_by_source.items()},
        )
        canonical_frames.append(current_frame)
        tracked_masks[0] = current_frame.label_image
        lineage_state: dict[int, list[int]] = {
            track_id: [0, 0, 0]
            for track_id in current_frame.raw_label_ids
        }
        next_track_id = max(current_frame.raw_label_ids, default=0) + 1
        LOGGER.info("Frame 0 initialization produced %s track(s).", current_frame.object_count)
        persist_tracking_progress(
            output_dir=output_dir,
            config=config,
            source_names=source_names,
            frame_count=frame_count,
            reference_shape=reference_shape,
            completed_frame=0,
            frame_mask=current_frame.label_image,
            lineage_state=lineage_state,
            next_track_id=next_track_id,
        )
        start_frame = 0
    else:
        canonical_frames = resume_state.canonical_frames
        tracked_masks = resume_state.tracked_masks
        current_frame = canonical_frames[-1]
        lineage_state = resume_state.lineage_state
        next_track_id = resume_state.next_track_id
        start_frame = resume_state.completed_frame
        if start_frame >= frame_count - 1:
            LOGGER.info("Checkpoint already covers the final frame; loading finished tracking result.")
            return build_tracking_result(source_names, canonical_frames, tracked_masks, lineage_state)

    for frame_index in range(start_frame, frame_count - 1):
        next_frames = {name: frames[frame_index + 1] for name, frames in frames_by_source.items()}
        current_frame, next_track_id = solve_frame_pair(
            config=config,
            frame_index=frame_index,
            current_frame=current_frame,
            next_frames=next_frames,
            next_raw_frame=raw_frames[frame_index + 1],
            scorers=scorers,
            lineage_state=lineage_state,
            next_track_id=next_track_id,
        )
        if len(canonical_frames) > frame_index + 1:
            canonical_frames[frame_index + 1] = current_frame
        else:
            canonical_frames.append(current_frame)
        tracked_masks[frame_index + 1] = current_frame.label_image
        persist_tracking_progress(
            output_dir=output_dir,
            config=config,
            source_names=source_names,
            frame_count=frame_count,
            reference_shape=reference_shape,
            completed_frame=frame_index + 1,
            frame_mask=current_frame.label_image,
            lineage_state=lineage_state,
            next_track_id=next_track_id,
        )

    return build_tracking_result(source_names, canonical_frames, tracked_masks, lineage_state)


def maybe_resume_tracking(
    config: TrackingConfig,
    raw_frames: np.ndarray,
    output_dir: Path,
    source_names: tuple[str, ...],
    frame_count: int,
    reference_shape: tuple[int, int],
) -> ResumeState | None:
    if config.force_retrack:
        LOGGER.info("Ignoring any saved tracking checkpoint because --force-retrack was requested.")
        return None

    checkpoint = projectio.load_tracking_checkpoint(output_dir)
    if checkpoint is None:
        LOGGER.info("No saved tracking checkpoint found; starting from frame 0.")
        return None

    if not checkpoint_matches(
        checkpoint=checkpoint,
        config=config,
        source_names=source_names,
        frame_count=frame_count,
        reference_shape=reference_shape,
    ):
        LOGGER.info("Saved tracking checkpoint does not match the current run; starting from frame 0.")
        return None

    completed_frame = checkpoint.completed_frame
    if completed_frame < 0 or completed_frame >= frame_count:
        LOGGER.info("Saved tracking checkpoint has an invalid completed frame; starting from frame 0.")
        return None

    tracked_masks = np.zeros((frame_count, *reference_shape), dtype=np.uint16)
    canonical_frames: list[FrameObjects] = []
    for frame_index in range(completed_frame + 1):
        mask = np.asarray(projectio.load_tracking_mask(output_dir, frame_index), dtype=np.uint16)
        if tuple(mask.shape) != reference_shape:
            raise ValueError(
                f"Saved tracked mask for frame {frame_index} has shape {mask.shape}, expected {reference_shape}."
            )
        tracked_masks[frame_index] = mask
        canonical_frames.append(projectio.build_frame_objects("tracked", frame_index, mask, raw_frames[frame_index]))

    LOGGER.info("Resuming tracking from saved frame %03d.", completed_frame)
    return ResumeState(
        completed_frame=completed_frame,
        canonical_frames=canonical_frames,
        tracked_masks=tracked_masks,
        lineage_state={
            track_id: [begin, end, parent]
            for track_id, (begin, end, parent) in checkpoint.lineage_state.items()
        },
        next_track_id=checkpoint.next_track_id,
    )


def checkpoint_matches(
    checkpoint: TrackingCheckpoint,
    config: TrackingConfig,
    source_names: tuple[str, ...],
    frame_count: int,
    reference_shape: tuple[int, int],
) -> bool:
    return (
        checkpoint.version == CHECKPOINT_VERSION
        and checkpoint.dataset_root == str(config.dataset_root)
        and checkpoint.track_sequence == config.track_sequence
        and checkpoint.seg_source == config.seg_source
        and checkpoint.selected_sources == source_names
        and checkpoint.frame_count == frame_count
        and checkpoint.frame_shape == reference_shape
        and abs(checkpoint.max_distance - config.max_distance) < 1e-9
        and abs(checkpoint.segmentation_reward - config.segmentation_reward) < 1e-9
    )


def persist_tracking_progress(
    output_dir,
    config: TrackingConfig,
    source_names: tuple[str, ...],
    frame_count: int,
    reference_shape: tuple[int, int],
    completed_frame: int,
    frame_mask: np.ndarray,
    lineage_state: dict[int, list[int]],
    next_track_id: int,
) -> None:
    projectio.write_tracking_mask(output_dir, completed_frame, frame_mask)
    projectio.write_lineage_rows(output_dir, lineage_rows_from_state(lineage_state))
    projectio.write_tracking_checkpoint(
        output_dir,
        TrackingCheckpoint(
            version=CHECKPOINT_VERSION,
            dataset_root=str(config.dataset_root),
            track_sequence=config.track_sequence,
            seg_source=config.seg_source,
            selected_sources=source_names,
            frame_count=frame_count,
            frame_shape=reference_shape,
            completed_frame=completed_frame,
            next_track_id=next_track_id,
            max_distance=config.max_distance,
            segmentation_reward=config.segmentation_reward,
            lineage_state={
                track_id: (begin, end, parent)
                for track_id, (begin, end, parent) in lineage_state.items()
            },
        ),
    )


def build_tracking_result(
    source_names: tuple[str, ...],
    canonical_frames: list[FrameObjects],
    tracked_masks: np.ndarray,
    lineage_state: dict[int, list[int]],
) -> TrackingResult:
    lineage_rows = lineage_rows_from_state(lineage_state)
    tracklets = build_tracklets(canonical_frames)
    LOGGER.info(
        "Decoded pairwise tracking with %s tracklet point(s) and %s lineage row(s).",
        len(tracklets),
        len(lineage_rows),
    )
    return TrackingResult(
        selected_sources=source_names,
        tracklets=tracklets,
        lineage_rows=lineage_rows,
        tracked_masks=tracked_masks,
    )


def lineage_rows_from_state(lineage_state: dict[int, list[int]]) -> tuple[LineageRecord, ...]:
    return tuple(
        LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent)
        for track_id, (begin, end, parent) in sorted(lineage_state.items())
    )


def initialize_first_frame(
    config: TrackingConfig,
    raw_frame: np.ndarray,
    first_frames_by_source: dict[str, FrameObjects],
) -> FrameObjects:
    if len(first_frames_by_source) == 1:
        source_name, frame = next(iter(first_frames_by_source.items()))
        LOGGER.info(
            "Initializing frame 0 directly from source '%s' with %s object(s).",
            source_name,
            frame.object_count,
        )
        selected_keys = [(source_name, object_index) for object_index in range(frame.object_count)]
    else:
        LOGGER.info(
            "Initializing frame 0 from %s source(s) with overlap-aware hypothesis selection.",
            len(first_frames_by_source),
        )
        selected_keys = select_initial_objects(config, first_frames_by_source)

    selected_keys.sort(key=lambda key: object_sort_key(first_frames_by_source, key[0], key[1]))
    assignments = [
        (source_name, object_index, track_id)
        for track_id, (source_name, object_index) in enumerate(selected_keys, start=1)
    ]
    mask = paint_track_mask(raw_frame.shape, first_frames_by_source, assignments)
    return projectio.build_frame_objects("tracked", 0, mask, raw_frame)


def select_initial_objects(
    config: TrackingConfig,
    first_frames_by_source: dict[str, FrameObjects],
) -> list[tuple[str, int]]:
    total_objects = sum(frame.object_count for frame in first_frames_by_source.values())
    if total_objects == 0:
        return []

    model = gp.Model("PyTr2dInit")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    activation_vars: dict[NextKey, gp.Var] = {}
    objective_terms: list[gp.LinExpr] = []

    for source_name, frame in first_frames_by_source.items():
        for object_index in range(frame.object_count):
            key = (source_name, object_index)
            activation = model.addVar(vtype=GRB.BINARY, name=f"init[{source_name},{object_index}]")
            activation_vars[key] = activation
            objective_terms.append(config.segmentation_reward * activation)

    overlap_constraint_count = 0
    for source_name_1, source_name_2 in combinations(first_frames_by_source, 2):
        frame_1 = first_frames_by_source[source_name_1]
        frame_2 = first_frames_by_source[source_name_2]
        for object_index_1, object_index_2 in overlapping_object_pairs(frame_1, frame_2):
            model.addConstr(
                activation_vars[(source_name_1, object_index_1)] + activation_vars[(source_name_2, object_index_2)] <= 1,
                name=f"init_overlap[{source_name_1},{source_name_2},{object_index_1},{object_index_2}]",
            )
            overlap_constraint_count += 1

    LOGGER.info(
        "Frame 0 initialization ILP: activation=%s, overlap constraints=%s.",
        len(activation_vars),
        overlap_constraint_count,
    )
    model.setObjective(quicksum(objective_terms), GRB.MINIMIZE)
    LOGGER.info("Starting Gurobi optimization for frame 0 initialization.")
    model.optimize()
    _assert_usable_status(model)

    selected = [key for key, variable in activation_vars.items() if variable.X > 0.5]
    LOGGER.info("Frame 0 initialization selected %s object(s).", len(selected))
    return selected


def solve_frame_pair(
    config: TrackingConfig,
    frame_index: int,
    current_frame: FrameObjects,
    next_frames: dict[str, FrameObjects],
    next_raw_frame: np.ndarray,
    scorers: EventScorers,
    lineage_state: dict[int, list[int]],
    next_track_id: int,
) -> tuple[FrameObjects, int]:
    next_object_count = sum(frame.object_count for frame in next_frames.values())
    LOGGER.info(
        "Solving pair %03d -> %03d with %s current track(s) and %s next candidate object(s).",
        frame_index,
        frame_index + 1,
        current_frame.object_count,
        next_object_count,
    )
    if current_frame.object_count == 0 and next_object_count == 0:
        LOGGER.info("Pair %03d -> %03d is empty; carrying forward an empty tracked frame.", frame_index, frame_index + 1)
        return projectio.build_frame_objects("tracked", frame_index + 1, np.zeros_like(next_raw_frame, dtype=np.uint16), next_raw_frame), next_track_id

    model = gp.Model(f"PyTr2dPair_{frame_index:03d}_{frame_index + 1:03d}")
    model.Params.OutputFlag = 1
    model.Params.LogToConsole = 1
    if config.log_file is not None:
        model.Params.LogFile = str(config.log_file)

    activation_vars: dict[NextKey, gp.Var] = {}
    appearance_vars: dict[NextKey, gp.Var] = {}
    disappearance_vars: dict[CurrentKey, gp.Var] = {}
    move_vars: dict[MoveKey, gp.Var] = {}
    division_vars: dict[DivisionKey, gp.Var] = {}
    outgoing_terms: dict[CurrentKey, list[gp.Var]] = defaultdict(list)
    incoming_terms: dict[NextKey, list[gp.Var]] = defaultdict(list)
    objective_terms: list[gp.LinExpr] = []

    candidate_maps = {
        source_name: candidate_neighborhoods(current_frame, next_frame, config.max_distance)
        for source_name, next_frame in next_frames.items()
    }

    for source_name, next_frame in next_frames.items():
        for object_index in range(next_frame.object_count):
            next_key = (source_name, object_index)
            activation = model.addVar(vtype=GRB.BINARY, name=f"act[{source_name},{frame_index + 1},{object_index}]")
            appearance = model.addVar(vtype=GRB.BINARY, name=f"app[{source_name},{frame_index + 1},{object_index}]")
            activation_vars[next_key] = activation
            appearance_vars[next_key] = appearance
            objective_terms.append(config.segmentation_reward * activation)
            objective_terms.append(scorers.appearance_cost(next_frame, object_index) * appearance)

    for current_index in range(current_frame.object_count):
        disappearance = model.addVar(vtype=GRB.BINARY, name=f"dis[{frame_index},{current_index}]")
        disappearance_vars[current_index] = disappearance
        objective_terms.append(scorers.disappearance_cost(current_frame, current_index) * disappearance)

        for source_name, next_frame in next_frames.items():
            candidate_indices = candidate_maps[source_name][current_index]
            for next_index in candidate_indices:
                move_key = (current_index, source_name, next_index)
                move = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"move[{frame_index},{current_index},{source_name},{next_index}]",
                )
                move_vars[move_key] = move
                outgoing_terms[current_index].append(move)
                incoming_terms[(source_name, next_index)].append(move)
                objective_terms.append(scorers.move_cost(current_frame, current_index, next_frame, next_index) * move)

            for child_index_1, child_index_2 in combinations(candidate_indices, 2):
                division_key = (current_index, source_name, child_index_1, child_index_2)
                division = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"div[{frame_index},{current_index},{source_name},{child_index_1},{child_index_2}]",
                )
                division_vars[division_key] = division
                outgoing_terms[current_index].append(division)
                incoming_terms[(source_name, child_index_1)].append(division)
                incoming_terms[(source_name, child_index_2)].append(division)
                objective_terms.append(
                    scorers.division_cost(current_frame, current_index, next_frame, child_index_1, child_index_2) * division
                )

    constraint_count = 0
    for current_index in range(current_frame.object_count):
        model.addConstr(
            quicksum(outgoing_terms[current_index]) + disappearance_vars[current_index] == 1,
            name=f"outgoing[{frame_index},{current_index}]",
        )
        constraint_count += 1

    for next_key, activation in activation_vars.items():
        model.addConstr(
            quicksum(incoming_terms[next_key]) + appearance_vars[next_key] == activation,
            name=f"incoming[{frame_index + 1},{next_key[0]},{next_key[1]}]",
        )
        constraint_count += 1

    overlap_constraint_count = 0
    for source_name_1, source_name_2 in combinations(next_frames, 2):
        frame_1 = next_frames[source_name_1]
        frame_2 = next_frames[source_name_2]
        for object_index_1, object_index_2 in overlapping_object_pairs(frame_1, frame_2):
            model.addConstr(
                activation_vars[(source_name_1, object_index_1)] + activation_vars[(source_name_2, object_index_2)] <= 1,
                name=f"overlap[{frame_index + 1},{source_name_1},{source_name_2},{object_index_1},{object_index_2}]",
            )
            overlap_constraint_count += 1

    LOGGER.info(
        "Pair %03d -> %03d ILP: activation=%s, appearance=%s, disappearance=%s, move=%s, division=%s, flow constraints=%s, overlap constraints=%s.",
        frame_index,
        frame_index + 1,
        len(activation_vars),
        len(appearance_vars),
        len(disappearance_vars),
        len(move_vars),
        len(division_vars),
        constraint_count,
        overlap_constraint_count,
    )
    model.setObjective(quicksum(objective_terms), GRB.MINIMIZE)
    LOGGER.info("Starting Gurobi optimization for pair %03d -> %03d.", frame_index, frame_index + 1)
    model.optimize()
    _assert_usable_status(model)
    LOGGER.info(
        "Finished pair %03d -> %03d optimization with status %s and objective %.6f.",
        frame_index,
        frame_index + 1,
        model.Status,
        float(model.ObjVal),
    )

    selected_appearances = {
        next_key
        for next_key, variable in appearance_vars.items()
        if variable.X > 0.5
    }
    selected_disappearances = {
        current_index
        for current_index, variable in disappearance_vars.items()
        if variable.X > 0.5
    }
    selected_moves = {
        current_index: (source_name, next_index)
        for (current_index, source_name, next_index), variable in move_vars.items()
        if variable.X > 0.5
    }
    selected_divisions = {
        current_index: ((source_name, child_index_1), (source_name, child_index_2))
        for (current_index, source_name, child_index_1, child_index_2), variable in division_vars.items()
        if variable.X > 0.5
    }

    assignments: list[tuple[str, int, int]] = []
    next_key_to_track_id: dict[NextKey, int] = {}

    for current_index in range(current_frame.object_count):
        track_id = int(current_frame.raw_label_ids[current_index])
        if current_index in selected_moves:
            next_key = selected_moves[current_index]
            next_key_to_track_id[next_key] = track_id
            assignments.append((next_key[0], next_key[1], track_id))
            lineage_state[track_id][1] = frame_index + 1

        if current_index in selected_divisions:
            child_keys = sorted(
                selected_divisions[current_index],
                key=lambda key: object_sort_key(next_frames, key[0], key[1]),
            )
            for child_key in child_keys:
                child_track_id = next_track_id
                next_track_id += 1
                next_key_to_track_id[child_key] = child_track_id
                assignments.append((child_key[0], child_key[1], child_track_id))
                lineage_state[child_track_id] = [frame_index + 1, frame_index + 1, track_id]

    for next_key in sorted(selected_appearances, key=lambda key: object_sort_key(next_frames, key[0], key[1])):
        if next_key in next_key_to_track_id:
            continue
        track_id = next_track_id
        next_track_id += 1
        next_key_to_track_id[next_key] = track_id
        assignments.append((next_key[0], next_key[1], track_id))
        lineage_state[track_id] = [frame_index + 1, frame_index + 1, 0]

    next_mask = paint_track_mask(next_raw_frame.shape, next_frames, assignments)
    next_frame = projectio.build_frame_objects("tracked", frame_index + 1, next_mask, next_raw_frame)
    LOGGER.info(
        "Pair %03d -> %03d decoded: moves=%s, divisions=%s, appearances=%s, disappearances=%s, cumulative tracks=%s.",
        frame_index,
        frame_index + 1,
        len(selected_moves),
        len(selected_divisions),
        len(selected_appearances),
        len(selected_disappearances),
        len(lineage_state),
    )
    return next_frame, next_track_id


def build_tracklets(canonical_frames: list[FrameObjects]) -> tuple[tuple[int, int, float, float], ...]:
    tracklets: list[tuple[int, int, float, float]] = []
    for frame in canonical_frames:
        for object_index, track_id in enumerate(frame.raw_label_ids):
            centroid_row, centroid_col = frame.centroids[object_index]
            tracklets.append((int(track_id), int(frame.frame_index), centroid_row, centroid_col))
    tracklets.sort(key=lambda row: (row[0], row[1]))
    return tuple(tracklets)


def paint_track_mask(
    shape: tuple[int, int],
    frames_by_source: dict[str, FrameObjects],
    assignments: list[tuple[str, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint16)
    for source_name, object_index, track_id in assignments:
        coords = frames_by_source[source_name].coords[object_index]
        existing_labels = mask[coords[:, 0], coords[:, 1]]
        conflicting_labels = existing_labels[(existing_labels != 0) & (existing_labels != track_id)]
        if conflicting_labels.size > 0:
            raise ValueError(
                f"Overlapping assignments detected while painting track {track_id} from source '{source_name}'."
            )
        mask[coords[:, 0], coords[:, 1]] = np.uint16(track_id)
    return mask


def object_sort_key(
    frames_by_source: dict[str, FrameObjects],
    source_name: str,
    object_index: int,
) -> tuple[float, float, str, int]:
    centroid_row, centroid_col = frames_by_source[source_name].centroids[object_index]
    return (centroid_row, centroid_col, source_name, object_index)


def overlapping_object_pairs(
    frame_1: FrameObjects,
    frame_2: FrameObjects,
) -> set[tuple[int, int]]:
    overlap_mask = (frame_1.label_image > 0) & (frame_2.label_image > 0)
    if not np.any(overlap_mask):
        return set()

    overlap_pairs = np.stack(
        [frame_1.label_image[overlap_mask], frame_2.label_image[overlap_mask]],
        axis=1,
    )
    unique_pairs = np.unique(overlap_pairs, axis=0)
    return {
        (
            frame_1.raw_label_to_index[int(raw_label_1)],
            frame_2.raw_label_to_index[int(raw_label_2)],
        )
        for raw_label_1, raw_label_2 in unique_pairs
    }


def _assert_usable_status(model: gp.Model) -> None:
    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi failed to find a usable solution. Status code: {model.Status}")


def _validate_frames(
    raw_frames: np.ndarray,
    frames_by_source: dict[str, list[FrameObjects]],
    frame_count: int,
    reference_shape: tuple[int, int],
) -> None:
    if raw_frames.ndim != 3:
        raise ValueError(f"Expected raw_frames to be a 3D stack, found shape {raw_frames.shape}.")
    for source_name, frames in frames_by_source.items():
        if len(frames) != frame_count:
            raise ValueError(f"Source '{source_name}' has {len(frames)} frames, expected {frame_count}.")
        for frame in frames:
            if frame.shape != reference_shape:
                raise ValueError(
                    f"Source '{source_name}' frame {frame.frame_index} has shape {frame.shape}, expected {reference_shape}."
                )
