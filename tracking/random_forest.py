from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, distance
from sklearn.ensemble import RandomForestClassifier

from tracking.types import FrameObjects, LineageRecord, TrackingConfig


LOGGER = logging.getLogger(__name__)

FEATURE_VERSION = "rf-events-v1"
MODEL_FILENAME = "event_scorers.pkl"


@dataclass(slots=True)
class EventScorers:
    move_model: RandomForestClassifier
    division_model: RandomForestClassifier
    appearance_model: RandomForestClassifier
    disappearance_model: RandomForestClassifier

    def move_cost(
        self,
        current_frame: FrameObjects,
        current_index: int,
        next_frame: FrameObjects,
        next_index: int,
    ) -> float:
        return probability_to_cost(
            self.move_model,
            movement_features(current_frame, current_index, next_frame, next_index),
        )

    def division_cost(
        self,
        current_frame: FrameObjects,
        current_index: int,
        next_frame: FrameObjects,
        child_index_1: int,
        child_index_2: int,
    ) -> float:
        return probability_to_cost(
            self.division_model,
            division_features(current_frame, current_index, next_frame, child_index_1, child_index_2),
        )

    def appearance_cost(self, frame: FrameObjects, object_index: int) -> float:
        return probability_to_cost(self.appearance_model, appearance_features(frame, object_index))

    def disappearance_cost(self, frame: FrameObjects, object_index: int) -> float:
        return probability_to_cost(self.disappearance_model, disappearance_features(frame, object_index))


@dataclass(slots=True)
class RandomForestEventTrainer:
    max_distance: float = 50.0
    train_iou_threshold: float = 0.3
    max_depth: int = 10
    random_state: int = 0
    n_jobs: int = 1

    def bundle_metadata(
        self,
        dataset_name: str,
        train_sequence: str,
    ) -> dict[str, Any]:
        return {
            "dataset_name": dataset_name,
            "train_sequence": train_sequence,
            "feature_version": FEATURE_VERSION,
            "max_distance": float(self.max_distance),
            "train_iou_threshold": float(self.train_iou_threshold),
            "max_depth": int(self.max_depth),
            "random_state": int(self.random_state),
            "n_jobs": int(self.n_jobs),
        }

    def fit(
        self,
        training_frames_by_source: dict[str, list[FrameObjects]],
        gt_frames: list[FrameObjects],
        lineage_records: dict[int, LineageRecord],
    ) -> EventScorers:
        LOGGER.info(
            "Training event scorers from %s training source(s) and %s GT frame(s).",
            len(training_frames_by_source),
            len(gt_frames),
        )
        move_rows: list[tuple[float, ...]] = []
        move_labels: list[int] = []
        division_rows: list[tuple[float, ...]] = []
        division_labels: list[int] = []
        appearance_rows: list[tuple[float, ...]] = []
        appearance_labels: list[int] = []
        disappearance_rows: list[tuple[float, ...]] = []
        disappearance_labels: list[int] = []

        division_targets = build_division_targets(lineage_records)

        for source_name, source_frames in training_frames_by_source.items():
            LOGGER.info("Aligning training source '%s' to GT.", source_name)
            frame_matches = align_source_to_gt_frames(source_frames, gt_frames, self.train_iou_threshold)
            source_examples = collect_training_examples(
                source_frames=source_frames,
                frame_matches=frame_matches,
                lineage_records=lineage_records,
                division_targets=division_targets,
                max_distance=self.max_distance,
            )
            LOGGER.info(
                "Collected training rows for '%s': move=%s, division=%s, appearance=%s, disappearance=%s.",
                source_name,
                len(source_examples[0]),
                len(source_examples[2]),
                len(source_examples[4]),
                len(source_examples[6]),
            )
            move_rows.extend(source_examples[0])
            move_labels.extend(source_examples[1])
            division_rows.extend(source_examples[2])
            division_labels.extend(source_examples[3])
            appearance_rows.extend(source_examples[4])
            appearance_labels.extend(source_examples[5])
            disappearance_rows.extend(source_examples[6])
            disappearance_labels.extend(source_examples[7])

        LOGGER.info(
            "Total training rows: move=%s, division=%s, appearance=%s, disappearance=%s.",
            len(move_rows),
            len(division_rows),
            len(appearance_rows),
            len(disappearance_rows),
        )
        return EventScorers(
            move_model=self._fit_classifier(move_rows, move_labels, "move"),
            division_model=self._fit_classifier(division_rows, division_labels, "division"),
            appearance_model=self._fit_classifier(appearance_rows, appearance_labels, "appearance"),
            disappearance_model=self._fit_classifier(disappearance_rows, disappearance_labels, "disappearance"),
        )

    def _fit_classifier(
        self,
        rows: list[tuple[float, ...]],
        labels: list[int],
        name: str,
    ) -> RandomForestClassifier:
        if not rows:
            raise ValueError(f"No training rows were generated for the {name} classifier.")
        if len(set(labels)) < 2:
            raise ValueError(f"The {name} classifier needs both positive and negative examples.")
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        LOGGER.info(
            "Fitting %s classifier with %s row(s): %s positive, %s negative.",
            name,
            len(rows),
            positive_count,
            negative_count,
        )
        model = RandomForestClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        model.fit(np.asarray(rows, dtype=float), np.asarray(labels, dtype=int))
        LOGGER.info("Finished fitting %s classifier.", name)
        return model


def resolve_model_path(config: TrackingConfig) -> Path:
    model_root = config.model_dir if config.model_dir is not None else Path("models")
    return model_root / config.dataset_root.name / config.train_sequence / MODEL_FILENAME


def save_event_scorers(
    path: Path,
    scorers: EventScorers,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata),
        "models": {
            "move_model": scorers.move_model,
            "division_model": scorers.division_model,
            "appearance_model": scorers.appearance_model,
            "disappearance_model": scorers.disappearance_model,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_event_scorers(
    path: Path,
    expected_metadata: dict[str, Any],
) -> EventScorers:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    metadata = payload.get("metadata")
    models = payload.get("models")
    if not isinstance(metadata, dict) or not isinstance(models, dict):
        raise ValueError(f"Invalid scorer bundle structure in {path}.")
    if metadata != expected_metadata:
        raise ValueError(f"Bundle metadata mismatch: expected {expected_metadata}, found {metadata}.")

    required_model_names = {
        "move_model",
        "division_model",
        "appearance_model",
        "disappearance_model",
    }
    if set(models) != required_model_names:
        raise ValueError(f"Invalid scorer bundle model set in {path}.")

    return EventScorers(
        move_model=models["move_model"],
        division_model=models["division_model"],
        appearance_model=models["appearance_model"],
        disappearance_model=models["disappearance_model"],
    )


def movement_features(
    current_frame: FrameObjects,
    current_index: int,
    next_frame: FrameObjects,
    next_index: int,
) -> tuple[float, float, float, float, float]:
    return (
        current_frame.intensity_std[current_index],
        next_frame.intensity_std[next_index],
        float(distance.euclidean(current_frame.centroids[current_index], next_frame.centroids[next_index])),
        float(current_frame.areas[current_index]),
        float(next_frame.areas[next_index]),
    )


def division_features(
    current_frame: FrameObjects,
    current_index: int,
    next_frame: FrameObjects,
    child_index_1: int,
    child_index_2: int,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    parent_centroid = current_frame.centroids[current_index]
    child_centroid_1 = next_frame.centroids[child_index_1]
    child_centroid_2 = next_frame.centroids[child_index_2]
    return (
        current_frame.intensity_std[current_index],
        next_frame.intensity_std[child_index_1],
        next_frame.intensity_std[child_index_2],
        float(distance.euclidean(parent_centroid, child_centroid_1)),
        float(distance.euclidean(parent_centroid, child_centroid_2)),
        float(distance.euclidean(child_centroid_1, child_centroid_2)),
        float(current_frame.areas[current_index]),
        float(next_frame.areas[child_index_1]),
        float(next_frame.areas[child_index_2]),
    )


def appearance_features(frame: FrameObjects, object_index: int) -> tuple[float, float, float]:
    return (
        frame.intensity_std[object_index],
        frame.border_distance[object_index],
        float(frame.areas[object_index]),
    )


def disappearance_features(frame: FrameObjects, object_index: int) -> tuple[float, float, float]:
    return appearance_features(frame, object_index)


def probability_to_cost(model: RandomForestClassifier, features: tuple[float, ...]) -> float:
    probabilities = model.predict_proba(np.asarray([features], dtype=float))[0]
    class_to_index = {int(label): index for index, label in enumerate(model.classes_)}
    if 1 not in class_to_index:
        raise ValueError("Expected the classifier to have a positive class labelled as 1.")
    positive_probability = float(probabilities[class_to_index[1]])
    return float(-np.log(max(positive_probability, 1e-6)))


def candidate_neighborhoods(
    current_frame: FrameObjects,
    next_frame: FrameObjects,
    max_distance: float,
) -> dict[int, list[int]]:
    neighborhoods: dict[int, list[int]] = {index: [] for index in range(current_frame.object_count)}
    if current_frame.object_count == 0 or next_frame.object_count == 0:
        return neighborhoods

    current_tree = KDTree(np.asarray(current_frame.centroids, dtype=float))
    next_tree = KDTree(np.asarray(next_frame.centroids, dtype=float))
    raw_neighborhoods = current_tree.query_ball_tree(next_tree, max_distance)
    return {index: sorted(neighbors) for index, neighbors in enumerate(raw_neighborhoods)}


def align_source_to_gt_frames(
    source_frames: list[FrameObjects],
    gt_frames: list[FrameObjects],
    iou_threshold: float,
) -> list[dict[int, int]]:
    if len(source_frames) != len(gt_frames):
        raise ValueError("Source frames and GT frames must have the same length for alignment.")
    return [align_frame_to_gt(source_frame, gt_frame, iou_threshold) for source_frame, gt_frame in zip(source_frames, gt_frames, strict=True)]


def align_frame_to_gt(
    source_frame: FrameObjects,
    gt_frame: FrameObjects,
    iou_threshold: float,
) -> dict[int, int]:
    if source_frame.object_count == 0 or gt_frame.object_count == 0:
        return {}

    intersections = intersection_areas(source_frame, gt_frame)
    cost_matrix = np.ones((source_frame.object_count, gt_frame.object_count), dtype=float)
    for (source_label, gt_label), intersection in intersections.items():
        source_index = source_frame.raw_label_to_index[source_label]
        gt_index = gt_frame.raw_label_to_index[gt_label]
        union = source_frame.areas[source_index] + gt_frame.areas[gt_index] - intersection
        iou = 0.0 if union == 0 else float(intersection / union)
        cost_matrix[source_index, gt_index] = 1.0 - iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: dict[int, int] = {}
    for source_index, gt_index in zip(row_ind, col_ind, strict=True):
        iou = 1.0 - cost_matrix[source_index, gt_index]
        if iou >= iou_threshold:
            matches[source_index] = gt_frame.raw_label_ids[gt_index]
    return matches


def intersection_areas(
    left_frame: FrameObjects,
    right_frame: FrameObjects,
) -> dict[tuple[int, int], int]:
    overlap_mask = (left_frame.label_image > 0) & (right_frame.label_image > 0)
    if not np.any(overlap_mask):
        return {}

    overlap_pairs = np.stack(
        [left_frame.label_image[overlap_mask], right_frame.label_image[overlap_mask]],
        axis=1,
    )
    unique_pairs, counts = np.unique(overlap_pairs, axis=0, return_counts=True)
    return {
        (int(pair[0]), int(pair[1])): int(count)
        for pair, count in zip(unique_pairs, counts, strict=True)
    }


def build_division_targets(lineage_records: dict[int, LineageRecord]) -> dict[tuple[int, int], tuple[int, ...]]:
    targets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for record in lineage_records.values():
        if record.parent > 0:
            targets[(record.parent, record.begin - 1)].append(record.track_id)
    return {
        key: tuple(sorted(children))
        for key, children in targets.items()
    }


def collect_training_examples(
    source_frames: list[FrameObjects],
    frame_matches: list[dict[int, int]],
    lineage_records: dict[int, LineageRecord],
    division_targets: dict[tuple[int, int], tuple[int, ...]],
    max_distance: float,
) -> tuple[
    list[tuple[float, ...]],
    list[int],
    list[tuple[float, ...]],
    list[int],
    list[tuple[float, ...]],
    list[int],
    list[tuple[float, ...]],
    list[int],
]:
    move_rows: list[tuple[float, ...]] = []
    move_labels: list[int] = []
    division_rows: list[tuple[float, ...]] = []
    division_labels: list[int] = []
    appearance_rows: list[tuple[float, ...]] = []
    appearance_labels: list[int] = []
    disappearance_rows: list[tuple[float, ...]] = []
    disappearance_labels: list[int] = []

    last_frame_index = len(source_frames) - 1
    for frame_index, frame in enumerate(source_frames):
        matches = frame_matches[frame_index]
        if frame_index > 0:
            for object_index in range(frame.object_count):
                appearance_rows.append(appearance_features(frame, object_index))
                gt_track_id = matches.get(object_index)
                is_positive = False
                if gt_track_id is not None:
                    record = lineage_records[gt_track_id]
                    is_positive = record.begin == frame_index and record.parent == 0
                appearance_labels.append(int(is_positive))

        if frame_index < last_frame_index:
            next_frame = source_frames[frame_index + 1]
            next_matches = frame_matches[frame_index + 1]
            neighborhoods = candidate_neighborhoods(frame, next_frame, max_distance)

            for object_index in range(frame.object_count):
                gt_track_id = matches.get(object_index)

                disappearance_rows.append(disappearance_features(frame, object_index))
                is_disappearance = False
                if gt_track_id is not None:
                    record = lineage_records[gt_track_id]
                    is_disappearance = record.end == frame_index and (gt_track_id, frame_index) not in division_targets
                disappearance_labels.append(int(is_disappearance))

                for next_index in neighborhoods[object_index]:
                    move_rows.append(movement_features(frame, object_index, next_frame, next_index))
                    is_move = gt_track_id is not None and next_matches.get(next_index) == gt_track_id
                    move_labels.append(int(is_move))

                for child_index_1, child_index_2 in combinations(neighborhoods[object_index], 2):
                    division_rows.append(division_features(frame, object_index, next_frame, child_index_1, child_index_2))
                    is_division = False
                    if gt_track_id is not None:
                        matched_children = (
                            next_matches.get(child_index_1),
                            next_matches.get(child_index_2),
                        )
                        if None not in matched_children:
                            positive_children = division_targets.get((gt_track_id, frame_index))
                            if positive_children is not None:
                                is_division = tuple(sorted(matched_children)) == positive_children
                    division_labels.append(int(is_division))

    return (
        move_rows,
        move_labels,
        division_rows,
        division_labels,
        appearance_rows,
        appearance_labels,
        disappearance_rows,
        disappearance_labels,
    )
