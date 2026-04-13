from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import shutil

import numpy as np
from skimage.measure import regionprops
import tifffile

from dataio import projectio
from tracking.types import LineageRecord


LOGGER = logging.getLogger(__name__)
EVENT_KIND_ORDER = (
    "continuation",
    "appearance",
    "disappearance",
    "division",
    "common",
    "embedseg_only",
    "stardist_only",
    "split_disagreement",
    "merge_disagreement",
)
EVENT_KIND_WEIGHTS = {
    "continuation": 1.0,
    "appearance": 4.0,
    "disappearance": 4.0,
    "division": 8.0,
    "common": 3.0,
    "embedseg_only": 4.0,
    "stardist_only": 4.0,
    "split_disagreement": 6.0,
    "merge_disagreement": 6.0,
}
DISTINCT_KIND_BONUS = 100.0
MAX_PER_KIND_BONUS = 3
GT_SEG_FRAME_BONUS = 25.0
DEFAULT_OUTPUT_SEQUENCE = "toy_consensus"
DEFAULT_SOURCE_NAMES = ("embedseg", "stardist")


@dataclass(slots=True, frozen=True)
class CropWindow:
    start_frame: int
    frame_count: int
    top: int
    left: int
    height: int
    width: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frame_count - 1

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


@dataclass(slots=True, frozen=True)
class SpatialEvent:
    kind: str
    frame_index: int
    bbox: tuple[int, int, int, int]
    description: str

    @property
    def center(self) -> tuple[float, float]:
        top, left, bottom, right = self.bbox
        return (0.5 * (top + bottom), 0.5 * (left + right))


@dataclass(slots=True, frozen=True)
class CropAssessment:
    valid: bool
    kept_track_ids: frozenset[int]
    disconnected_label_ids: tuple[int, ...]
    gapped_track_ids: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class SelectionResult:
    crop: CropWindow
    total_score: float
    temporal_score: float
    spatial_score: float
    counts_by_kind: dict[str, int]
    missing_kinds: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a compact spatiotemporal crop from the Fluo-N2DL-HeLa tracking sequence "
            "that preserves diverse GT and consensus-like events, then write a toy dataset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa",
        help="Dataset root containing the raw sequence and <sequence>_GT folders.",
    )
    parser.add_argument(
        "--extra-seg-root",
        default="./data/Fluo-N2DL-HeLa_train/Segmentations",
        help="External segmentation root containing embedseg and stardist folders.",
    )
    parser.add_argument(
        "--track-sequence",
        default="02",
        help="Sequence to crop into a toy subset.",
    )
    parser.add_argument(
        "--source-names",
        nargs=2,
        default=DEFAULT_SOURCE_NAMES,
        metavar=("SOURCE_1", "SOURCE_2"),
        help="The two external segmentation sources to compare when scoring consensus-like events.",
    )
    parser.add_argument(
        "--output-sequence",
        default=DEFAULT_OUTPUT_SEQUENCE,
        help="Name of the cropped sequence directory to create under --dataset-root.",
    )
    parser.add_argument(
        "--output-extra-seg-root",
        default=None,
        help="Optional output root for cropped external segmentations. Defaults to a sibling Segmentations_<sequence> folder.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=10,
        help="Number of consecutive frames in the toy subset.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=200,
        help="Height of the spatial crop.",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=300,
        help="Width of the spatial crop.",
    )
    parser.add_argument(
        "--agreement-iou-threshold",
        type=float,
        default=0.8,
        help="IoU threshold used to classify a one-to-one source overlap as a common event.",
    )
    parser.add_argument(
        "--overlap-iou-threshold",
        type=float,
        default=0.1,
        help="IoU threshold used to classify looser source overlaps for split and merge disagreements.",
    )
    parser.add_argument(
        "--min-bbox-coverage",
        type=float,
        default=0.5,
        help="Minimum event-bounding-box coverage required for a crop to count as covering that event.",
    )
    parser.add_argument(
        "--spatial-step",
        type=int,
        default=50,
        help="Grid step for crop-position search.",
    )
    parser.add_argument(
        "--temporal-top-k",
        type=int,
        default=8,
        help="How many top temporal windows to keep before running the spatial search.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and report the best crop without writing any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing output sequence and cropped segmentation folders.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Terminal logging verbosity.",
    )
    return parser


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def read_image(path: Path) -> np.ndarray:
    try:
        image = np.asarray(tifffile.imread(path))
    except ValueError as exc:
        lowered = str(exc).lower()
        if "imagecodecs" not in lowered and "compression" not in lowered:
            raise
        try:
            from PIL import Image
        except ModuleNotFoundError as pillow_exc:  # pragma: no cover - depends on environment.
            raise ValueError(
                f"Could not decode {path}. Install imagecodecs or Pillow to read compressed TIFF files."
            ) from pillow_exc
        with Image.open(path) as handle:
            image = np.asarray(handle)
    return np.asarray(image)


def load_stack(paths: tuple[Path, ...], *, normalize_source: bool = False) -> np.ndarray:
    if not paths:
        raise ValueError("Expected at least one TIFF path to load.")
    images: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        image = read_image(path)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image at {path}, found shape {image.shape}.")
        if normalize_source:
            image, _split_labels = projectio.normalize_source_label_image(image)
        shape = tuple(int(value) for value in image.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Inconsistent image shape under {path.parent}: saw {shape}, expected {expected_shape}.")
        images.append(np.asarray(image, dtype=np.uint16))
    return np.stack(images)


def load_indexed_stack(paths: tuple[Path, ...], *, normalize_source: bool = False) -> dict[int, np.ndarray]:
    if not paths:
        return {}
    images: dict[int, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        frame_index = projectio.extract_frame_index(path)
        if frame_index in images:
            raise ValueError(f"Duplicate frame index {frame_index} under {path.parent}.")
        image = read_image(path)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image at {path}, found shape {image.shape}.")
        if normalize_source:
            image, _split_labels = projectio.normalize_source_label_image(image)
        shape = tuple(int(value) for value in image.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Inconsistent image shape under {path.parent}: saw {shape}, expected {expected_shape}.")
        images[frame_index] = np.asarray(image, dtype=np.uint16)
    return images


def relabel_sequentially(label_image: np.ndarray) -> np.ndarray:
    relabeled = np.zeros(label_image.shape, dtype=np.uint32)
    positive_ids = sorted(int(value) for value in np.unique(label_image) if int(value) > 0)
    for next_label_id, label_id in enumerate(positive_ids, start=1):
        relabeled[label_image == label_id] = next_label_id
    if relabeled.size == 0 or int(np.max(relabeled)) <= np.iinfo(np.uint16).max:
        return relabeled.astype(np.uint16, copy=False)
    return relabeled


def label_bboxes_and_areas(label_image: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    areas: dict[int, int] = {}
    for prop in regionprops(label_image):
        boxes[int(prop.label)] = tuple(int(value) for value in prop.bbox)
        areas[int(prop.label)] = int(prop.area)
    return boxes, areas


def bbox_union(*boxes: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return None
    return (
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    )


def pair_intersection_counts(left: np.ndarray, right: np.ndarray) -> dict[tuple[int, int], int]:
    overlap_mask = (left > 0) & (right > 0)
    if not np.any(overlap_mask):
        return {}
    packed_pairs = (
        left[overlap_mask].astype(np.uint32, copy=False) << np.uint32(16)
    ) | right[overlap_mask].astype(np.uint32, copy=False)
    unique_pairs, counts = np.unique(packed_pairs, return_counts=True)
    return {
        (int(pair >> np.uint32(16)), int(pair & np.uint32(0xFFFF))): int(count)
        for pair, count in zip(unique_pairs, counts, strict=True)
        if int(pair >> np.uint32(16)) > 0 and int(pair & np.uint32(0xFFFF)) > 0
    }


def safe_iou(intersection: int, left_area: int, right_area: int) -> float:
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def extract_gt_events(gt_masks: np.ndarray, lineage_rows: dict[int, LineageRecord]) -> list[SpatialEvent]:
    frame_count = int(len(gt_masks))
    bboxes_by_frame = [label_bboxes_and_areas(frame)[0] for frame in gt_masks]
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for row in lineage_rows.values():
        if row.parent > 0:
            children_by_parent[row.parent].append(row.track_id)

    events: list[SpatialEvent] = []
    for track_id, row in sorted(lineage_rows.items()):
        if row.begin > 0:
            box = bboxes_by_frame[row.begin].get(track_id)
            if box is not None:
                events.append(SpatialEvent("appearance", row.begin, box, f"track {track_id} appears"))
        if row.end < frame_count - 1:
            box = bboxes_by_frame[row.end].get(track_id)
            if box is not None:
                events.append(SpatialEvent("disappearance", row.end, box, f"track {track_id} disappears"))
        for frame_index in range(row.begin, row.end):
            box = bbox_union(
                bboxes_by_frame[frame_index].get(track_id),
                bboxes_by_frame[frame_index + 1].get(track_id),
            )
            if box is not None:
                events.append(SpatialEvent("continuation", frame_index, box, f"track {track_id} continues"))

    for parent_id, children in sorted(children_by_parent.items()):
        if len(children) != 2:
            continue
        parent_row = lineage_rows[parent_id]
        child_ids = tuple(sorted(children))
        child_boxes = [bboxes_by_frame[parent_row.end + 1].get(child_id) for child_id in child_ids if parent_row.end + 1 < frame_count]
        box = bbox_union(
            bboxes_by_frame[parent_row.end].get(parent_id),
            *child_boxes,
        )
        if box is not None:
            events.append(
                SpatialEvent(
                    "division",
                    parent_row.end,
                    box,
                    f"track {parent_id} divides into {child_ids[0]} and {child_ids[1]}",
                )
            )
    return events


def extract_source_consensus_events(
    left_masks: np.ndarray,
    right_masks: np.ndarray,
    *,
    left_name: str,
    right_name: str,
    agreement_iou_threshold: float,
    overlap_iou_threshold: float,
) -> list[SpatialEvent]:
    events: list[SpatialEvent] = []
    for frame_index, (left_frame, right_frame) in enumerate(zip(left_masks, right_masks, strict=True)):
        if frame_index % 10 == 0:
            LOGGER.info(
                "Scanning source-overlap events: frame %s/%s.",
                frame_index + 1,
                len(left_masks),
            )
        left_boxes, left_areas = label_bboxes_and_areas(left_frame)
        right_boxes, right_areas = label_bboxes_and_areas(right_frame)
        intersections = pair_intersection_counts(left_frame, right_frame)
        left_neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
        right_neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
        left_best: dict[int, tuple[int, float]] = {}
        right_best: dict[int, tuple[int, float]] = {}
        common_candidates: list[tuple[int, int]] = []

        for (left_id, right_id), intersection in intersections.items():
            iou = safe_iou(intersection, left_areas[left_id], right_areas[right_id])
            if iou >= overlap_iou_threshold:
                left_neighbors[left_id].append((right_id, iou))
                right_neighbors[right_id].append((left_id, iou))
            previous_left = left_best.get(left_id)
            if previous_left is None or iou > previous_left[1]:
                left_best[left_id] = (right_id, iou)
            previous_right = right_best.get(right_id)
            if previous_right is None or iou > previous_right[1]:
                right_best[right_id] = (left_id, iou)
            if iou >= agreement_iou_threshold:
                common_candidates.append((left_id, right_id))

        common_pairs: set[tuple[int, int]] = set()
        for left_id, right_id in common_candidates:
            left_choice = left_best.get(left_id)
            right_choice = right_best.get(right_id)
            if left_choice is None or right_choice is None:
                continue
            if left_choice[0] == right_id and right_choice[0] == left_id:
                common_pairs.add((left_id, right_id))

        for left_id, right_id in sorted(common_pairs):
            box = bbox_union(left_boxes.get(left_id), right_boxes.get(right_id))
            if box is not None:
                events.append(
                    SpatialEvent(
                        "common",
                        frame_index,
                        box,
                        f"{left_name}:{left_id} matches {right_name}:{right_id}",
                    )
                )

        for left_id in sorted(left_boxes):
            if not left_neighbors.get(left_id):
                box = left_boxes.get(left_id)
                if box is not None:
                    events.append(
                        SpatialEvent(
                            "embedseg_only" if left_name == "embedseg" else f"{left_name}_only",
                            frame_index,
                            box,
                            f"{left_name}:{left_id} has no significant overlap",
                        )
                    )
            if len(left_neighbors.get(left_id, ())) >= 2:
                box = bbox_union(left_boxes.get(left_id), *(right_boxes.get(right_id) for right_id, _iou in left_neighbors[left_id]))
                if box is not None:
                    events.append(
                        SpatialEvent(
                            "split_disagreement",
                            frame_index,
                            box,
                            f"{left_name}:{left_id} overlaps multiple {right_name} objects",
                        )
                    )

        for right_id in sorted(right_boxes):
            if not right_neighbors.get(right_id):
                box = right_boxes.get(right_id)
                if box is not None:
                    events.append(
                        SpatialEvent(
                            "stardist_only" if right_name == "stardist" else f"{right_name}_only",
                            frame_index,
                            box,
                            f"{right_name}:{right_id} has no significant overlap",
                        )
                    )
            if len(right_neighbors.get(right_id, ())) >= 2:
                box = bbox_union(right_boxes.get(right_id), *(left_boxes.get(left_id) for left_id, _iou in right_neighbors[right_id]))
                if box is not None:
                    events.append(
                        SpatialEvent(
                            "merge_disagreement",
                            frame_index,
                            box,
                            f"{right_name}:{right_id} overlaps multiple {left_name} objects",
                        )
                    )
    return events


def event_counts(events: list[SpatialEvent], *, start_frame: int, end_frame: int) -> Counter[str]:
    return Counter(
        event.kind
        for event in events
        if start_frame <= event.frame_index <= end_frame and event.kind in EVENT_KIND_WEIGHTS
    )


def score_counts(counts: Counter[str]) -> float:
    score = DISTINCT_KIND_BONUS * len(counts)
    for kind, count in counts.items():
        score += EVENT_KIND_WEIGHTS.get(kind, 0.0) * min(int(count), MAX_PER_KIND_BONUS)
    return float(score)


def pick_temporal_windows(
    events: list[SpatialEvent],
    frame_count: int,
    window_frames: int,
    top_k: int,
    *,
    annotated_seg_frames: frozenset[int] = frozenset(),
) -> list[tuple[int, float]]:
    if window_frames > frame_count:
        raise ValueError(f"Requested {window_frames} frames, but the sequence only has {frame_count}.")
    ranked: list[tuple[int, float]] = []
    for start_frame in range(frame_count - window_frames + 1):
        counts = event_counts(events, start_frame=start_frame, end_frame=start_frame + window_frames - 1)
        if not counts:
            continue
        seg_frame_bonus = GT_SEG_FRAME_BONUS * sum(
            1 for frame_index in annotated_seg_frames if start_frame <= frame_index <= start_frame + window_frames - 1
        )
        ranked.append((start_frame, score_counts(counts) + seg_frame_bonus))
    if not ranked:
        raise RuntimeError("No temporal windows contained any scored events.")
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[: max(1, top_k)]


def crop_array(image: np.ndarray, crop: CropWindow) -> np.ndarray:
    return np.asarray(image[crop.top:crop.bottom, crop.left:crop.right])


def crop_covers_event(crop: CropWindow, event: SpatialEvent, *, min_bbox_coverage: float) -> bool:
    top, left, bottom, right = event.bbox
    intersection_top = max(top, crop.top)
    intersection_left = max(left, crop.left)
    intersection_bottom = min(bottom, crop.bottom)
    intersection_right = min(right, crop.right)
    if intersection_top >= intersection_bottom or intersection_left >= intersection_right:
        return False
    box_area = max(1, (bottom - top) * (right - left))
    overlap_area = (intersection_bottom - intersection_top) * (intersection_right - intersection_left)
    center_row, center_col = event.center
    center_inside = crop.top <= center_row < crop.bottom and crop.left <= center_col < crop.right
    return center_inside or (overlap_area / box_area) >= min_bbox_coverage


def assess_gt_crop(gt_masks: np.ndarray, crop: CropWindow) -> CropAssessment:
    kept_track_frames: dict[int, list[int]] = defaultdict(list)
    disconnected_label_ids: set[int] = set()
    for local_index, frame in enumerate(gt_masks[crop.start_frame:crop.end_frame + 1]):
        cropped = crop_array(frame, crop)
        disconnected = projectio.disconnected_label_components(cropped)
        disconnected_label_ids.update(label_id for label_id, _count in disconnected)
        for track_id in (int(value) for value in np.unique(cropped) if int(value) > 0):
            kept_track_frames[track_id].append(local_index)

    gapped_track_ids = sorted(
        track_id
        for track_id, frames in kept_track_frames.items()
        if frames and (frames[-1] - frames[0] + 1 != len(frames))
    )
    return CropAssessment(
        valid=not disconnected_label_ids and not gapped_track_ids,
        kept_track_ids=frozenset(kept_track_frames),
        disconnected_label_ids=tuple(sorted(disconnected_label_ids)),
        gapped_track_ids=tuple(gapped_track_ids),
    )


def candidate_starts(max_start: int, step: int) -> list[int]:
    starts = list(range(0, max_start + 1, step))
    if not starts:
        return [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def iter_crop_candidates(
    image_shape: tuple[int, int],
    *,
    start_frame: int,
    frame_count: int,
    crop_height: int,
    crop_width: int,
    spatial_step: int,
) -> list[CropWindow]:
    height, width = image_shape
    if crop_height > height or crop_width > width:
        raise ValueError(
            f"Requested crop size {(crop_height, crop_width)} exceeds image shape {(height, width)}."
        )
    top_starts = candidate_starts(height - crop_height, spatial_step)
    left_starts = candidate_starts(width - crop_width, spatial_step)
    return [
        CropWindow(
            start_frame=start_frame,
            frame_count=frame_count,
            top=top,
            left=left,
            height=crop_height,
            width=crop_width,
        )
        for top in top_starts
        for left in left_starts
    ]


def select_best_crop(
    events: list[SpatialEvent],
    gt_masks: np.ndarray,
    *,
    window_frames: int,
    crop_height: int,
    crop_width: int,
    spatial_step: int,
    temporal_top_k: int,
    min_bbox_coverage: float,
    annotated_seg_frames: frozenset[int] = frozenset(),
    available_st_frames: frozenset[int] = frozenset(),
) -> SelectionResult:
    frame_count = int(len(gt_masks))
    temporal_windows = pick_temporal_windows(
        events,
        frame_count,
        window_frames,
        temporal_top_k,
        annotated_seg_frames=annotated_seg_frames,
    )
    image_shape = tuple(int(value) for value in gt_masks[0].shape)
    best_result: SelectionResult | None = None

    LOGGER.info("Evaluating %s temporal candidate window(s).", len(temporal_windows))
    for window_index, (start_frame, temporal_score) in enumerate(temporal_windows, start=1):
        end_frame = start_frame + window_frames - 1
        relevant_events = [
            event
            for event in events
            if start_frame <= event.frame_index <= end_frame and event.kind in EVENT_KIND_WEIGHTS
        ]
        if not relevant_events:
            continue
        crop_candidates = iter_crop_candidates(
            image_shape,
            start_frame=start_frame,
            frame_count=window_frames,
            crop_height=crop_height,
            crop_width=crop_width,
            spatial_step=spatial_step,
        )
        LOGGER.info(
            "Temporal window %s/%s: frames %s..%s with %s relevant event(s) and %s crop candidate(s).",
            window_index,
            len(temporal_windows),
            start_frame,
            end_frame,
            len(relevant_events),
            len(crop_candidates),
        )

        for crop in crop_candidates:
            crop_frame_indices = frozenset(range(crop.start_frame, crop.end_frame + 1))
            if available_st_frames and not crop_frame_indices.issubset(available_st_frames):
                continue
            assessment = assess_gt_crop(gt_masks, crop)
            if not assessment.valid:
                continue
            counts = Counter(
                event.kind
                for event in relevant_events
                if crop_covers_event(crop, event, min_bbox_coverage=min_bbox_coverage)
            )
            if not counts:
                continue
            spatial_score = score_counts(counts)
            total_score = temporal_score + spatial_score
            missing_kinds = tuple(kind for kind in EVENT_KIND_ORDER if kind not in counts)
            candidate = SelectionResult(
                crop=crop,
                total_score=total_score,
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                counts_by_kind=dict(sorted(counts.items())),
                missing_kinds=missing_kinds,
            )
            if best_result is None or (candidate.total_score, -candidate.crop.start_frame, -candidate.crop.top, -candidate.crop.left) > (
                best_result.total_score,
                -best_result.crop.start_frame,
                -best_result.crop.top,
                -best_result.crop.left,
            ):
                best_result = candidate
                LOGGER.info(
                    "New best crop: frames %s..%s top=%s left=%s total_score=%.1f kinds=%s.",
                    candidate.crop.start_frame,
                    candidate.crop.end_frame,
                    candidate.crop.top,
                    candidate.crop.left,
                    candidate.total_score,
                    ", ".join(sorted(candidate.counts_by_kind)),
                )

    if best_result is None:
        raise RuntimeError(
            "No valid crop satisfied the GT continuity constraints. Try a larger crop, more frames, or a smaller spatial step."
        )
    return best_result


def rebuild_cropped_lineage_rows(
    gt_masks: np.ndarray,
    crop: CropWindow,
    lineage_rows: dict[int, LineageRecord],
) -> tuple[LineageRecord, ...]:
    assessment = assess_gt_crop(gt_masks, crop)
    if not assessment.valid:
        raise ValueError(
            "The selected crop produced invalid GT masks and cannot be converted to a valid lineage file."
        )
    track_frames: dict[int, list[int]] = defaultdict(list)
    for local_index, frame in enumerate(gt_masks[crop.start_frame:crop.end_frame + 1]):
        cropped = crop_array(frame, crop)
        for track_id in (int(value) for value in np.unique(cropped) if int(value) > 0):
            track_frames[track_id].append(local_index)

    kept_track_ids = frozenset(track_frames)
    rows: list[LineageRecord] = []
    for track_id in sorted(track_frames):
        original = lineage_rows[track_id]
        frames = track_frames[track_id]
        parent = original.parent if original.parent in kept_track_ids else 0
        if parent > 0 and parent in track_frames:
            parent_end = track_frames[parent][-1]
            if parent_end != frames[0] - 1:
                parent = 0
        rows.append(
            LineageRecord(
                track_id=track_id,
                begin=frames[0],
                end=frames[-1],
                parent=parent,
            )
        )
    return tuple(rows)


def resolve_output_extra_seg_root(dataset_root: Path, output_sequence: str, requested_path: str | None) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    return dataset_root.parent / f"Segmentations_{output_sequence}"


def ensure_clean_output_dirs(paths: list[Path], *, force: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        if not force:
            raise ValueError(f"Output path already exists: {path}. Use --force to overwrite it.")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def write_selection_outputs(
    *,
    dataset_root: Path,
    extra_seg_output_root: Path,
    output_sequence: str,
    crop: CropWindow,
    raw_paths: tuple[Path, ...],
    gt_masks: np.ndarray,
    gt_seg_masks_by_frame: dict[int, np.ndarray],
    st_seg_masks_by_frame: dict[int, np.ndarray],
    source_masks: dict[str, np.ndarray],
    lineage_rows: dict[int, LineageRecord],
    force: bool,
) -> tuple[Path, Path, Path]:
    raw_output_dir = dataset_root / output_sequence
    gt_output_root = dataset_root / f"{output_sequence}_GT"
    st_output_root = dataset_root / f"{output_sequence}_ST"
    gt_tra_dir = gt_output_root / "TRA"
    gt_seg_dir = gt_output_root / "SEG"
    st_seg_dir = st_output_root / "SEG"
    source_output_dirs = {source_name: extra_seg_output_root / source_name for source_name in source_masks}

    ensure_clean_output_dirs(
        [raw_output_dir, gt_output_root, st_output_root, extra_seg_output_root],
        force=force,
    )
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    gt_tra_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_dir.mkdir(parents=True, exist_ok=True)
    st_seg_dir.mkdir(parents=True, exist_ok=True)
    for output_dir in source_output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    written_gt_seg_frames: list[int] = []
    for local_index, source_frame_index in enumerate(range(crop.start_frame, crop.end_frame + 1)):
        raw_image = crop_array(read_image(raw_paths[source_frame_index]), crop)
        tifffile.imwrite(raw_output_dir / f"t{local_index:03d}.tif", np.asarray(raw_image, dtype=np.uint16))

        gt_crop = crop_array(gt_masks[source_frame_index], crop)
        projectio.validate_single_component_labels(gt_crop, gt_tra_dir / f"man_track{local_index:03d}.tif", kind="Cropped GT frame")
        tifffile.imwrite(gt_tra_dir / f"man_track{local_index:03d}.tif", np.asarray(gt_crop, dtype=np.uint16))

        gt_seg_mask = gt_seg_masks_by_frame.get(source_frame_index)
        if gt_seg_mask is not None:
            gt_seg_crop = crop_array(gt_seg_mask, crop)
            projectio.validate_single_component_labels(
                gt_seg_crop,
                gt_seg_dir / f"man_seg{local_index:03d}.tif",
                kind="Cropped GT segmentation frame",
            )
            tifffile.imwrite(gt_seg_dir / f"man_seg{local_index:03d}.tif", np.asarray(gt_seg_crop, dtype=np.uint16))
            written_gt_seg_frames.append(local_index)

        st_seg_mask = st_seg_masks_by_frame.get(source_frame_index)
        if st_seg_mask is None:
            raise ValueError(
                f"Missing ST segmentation mask for source frame {source_frame_index:03d}; "
                "the selected toy crop must only use frames present in the source ST annotations."
            )
        st_seg_crop = crop_array(st_seg_mask, crop)
        projectio.validate_single_component_labels(
            st_seg_crop,
            st_seg_dir / f"man_seg{local_index:03d}.tif",
            kind="Cropped ST segmentation frame",
        )
        tifffile.imwrite(st_seg_dir / f"man_seg{local_index:03d}.tif", np.asarray(st_seg_crop, dtype=np.uint16))

        for source_name, stack in source_masks.items():
            cropped_source = crop_array(stack[source_frame_index], crop)
            normalized_source, _split_labels = projectio.normalize_source_label_image(cropped_source)
            normalized_source = relabel_sequentially(normalized_source)
            tifffile.imwrite(
                source_output_dirs[source_name] / f"frame{local_index:03d}.tif",
                np.asarray(normalized_source, dtype=np.uint16),
            )

    cropped_rows = rebuild_cropped_lineage_rows(gt_masks, crop, lineage_rows)
    projectio.write_lineage_rows(gt_tra_dir, cropped_rows)
    projectio.write_lineage_rows(gt_tra_dir, cropped_rows, filename="man_track.txt")
    if not written_gt_seg_frames:
        raise ValueError(
            "The selected crop does not contain any GT SEG frames. Choose a window that overlaps the sparse gold segmentation annotations."
        )
    return raw_output_dir, gt_output_root, extra_seg_output_root


def build_report_payload(
    *,
    dataset_root: Path,
    extra_seg_root: Path,
    output_sequence: str,
    selection: SelectionResult,
    source_names: tuple[str, str],
    gt_seg_source_frames: tuple[int, ...],
    dry_run: bool,
) -> dict[str, object]:
    track_sequence_root = dataset_root / output_sequence
    gt_root = dataset_root / f"{output_sequence}_GT"
    st_root = dataset_root / f"{output_sequence}_ST"
    consensus_command = [
        "uv",
        "run",
        "python",
        "main.py",
        "--mode",
        "consensus",
        "--dataset-root",
        str(dataset_root),
        "--track-sequence",
        output_sequence,
        "--extra-seg-root",
        str(extra_seg_root),
        "--consensus-sources",
        source_names[0],
        source_names[1],
    ]
    return {
        "status": "dry_run" if dry_run else "written",
        "output_sequence": output_sequence,
        "track_sequence_dir": str(track_sequence_root),
        "gt_dir": str(gt_root),
        "st_dir": str(st_root),
        "extra_seg_root": str(extra_seg_root),
        "gt_seg_source_frames": [int(frame_index) for frame_index in gt_seg_source_frames],
        "gt_seg_local_frames": [int(frame_index - selection.crop.start_frame) for frame_index in gt_seg_source_frames],
        "selected_window": {
            "source_start_frame": int(selection.crop.start_frame),
            "source_end_frame": int(selection.crop.end_frame),
            "frame_count": int(selection.crop.frame_count),
            "top": int(selection.crop.top),
            "left": int(selection.crop.left),
            "height": int(selection.crop.height),
            "width": int(selection.crop.width),
        },
        "scores": {
            "total": float(selection.total_score),
            "temporal": float(selection.temporal_score),
            "spatial": float(selection.spatial_score),
        },
        "covered_event_counts": selection.counts_by_kind,
        "missing_event_kinds": list(selection.missing_kinds),
        "consensus_command": consensus_command,
    }


def format_report_text(payload: dict[str, object]) -> str:
    window = payload["selected_window"]
    counts = payload["covered_event_counts"]
    missing = payload["missing_event_kinds"]
    lines = [
        f"status: {payload['status']}",
        f"output_sequence: {payload['output_sequence']}",
        f"gt_dir: {payload['gt_dir']}",
        f"st_dir: {payload['st_dir']}",
        (
            "selected_window: "
            f"frames {window['source_start_frame']}..{window['source_end_frame']}, "
            f"crop top={window['top']} left={window['left']} "
            f"height={window['height']} width={window['width']}"
        ),
        "gt_seg_source_frames: " + (
            ", ".join(str(frame_index) for frame_index in payload["gt_seg_source_frames"])
            if payload["gt_seg_source_frames"]
            else "<none>"
        ),
        "gt_seg_local_frames: " + (
            ", ".join(str(frame_index) for frame_index in payload["gt_seg_local_frames"])
            if payload["gt_seg_local_frames"]
            else "<none>"
        ),
        (
            "scores: "
            f"total={payload['scores']['total']:.1f} "
            f"temporal={payload['scores']['temporal']:.1f} "
            f"spatial={payload['scores']['spatial']:.1f}"
        ),
        "covered_event_counts: " + ", ".join(f"{kind}={counts[kind]}" for kind in sorted(counts)),
        "missing_event_kinds: " + (", ".join(missing) if missing else "<none>"),
        "consensus_command: " + " ".join(str(token) for token in payload["consensus_command"]),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    extra_seg_root = Path(args.extra_seg_root).expanduser().resolve()
    source_names = tuple(str(value) for value in args.source_names)
    output_sequence = str(args.output_sequence)
    extra_seg_output_root = resolve_output_extra_seg_root(dataset_root, output_sequence, args.output_extra_seg_root)

    raw_dir = dataset_root / args.track_sequence
    raw_paths = projectio.sorted_tiff_paths(raw_dir)
    if not raw_paths:
        raise ValueError(f"No raw frames were found under {raw_dir}.")
    gt_dir = dataset_root / f"{args.track_sequence}_GT" / "TRA"
    gt_paths = projectio.sorted_tiff_paths(gt_dir)
    if len(gt_paths) != len(raw_paths):
        raise ValueError(
            f"GT tracking masks under {gt_dir} have {len(gt_paths)} frames, but raw sequence {raw_dir} has {len(raw_paths)}."
        )
    gt_seg_dir = dataset_root / f"{args.track_sequence}_GT" / "SEG"
    gt_seg_paths = projectio.sorted_tiff_paths(gt_seg_dir)
    st_seg_dir = dataset_root / f"{args.track_sequence}_ST" / "SEG"
    st_seg_paths = projectio.sorted_tiff_paths(st_seg_dir)
    if not st_seg_paths:
        raise ValueError(f"No ST segmentation masks were found under {st_seg_dir}.")
    source_paths = {
        source_name: projectio.sorted_tiff_paths(extra_seg_root / source_name)
        for source_name in source_names
    }
    for source_name, paths in source_paths.items():
        if len(paths) != len(raw_paths):
            raise ValueError(
                f"Source {source_name} under {extra_seg_root / source_name} has {len(paths)} frames, expected {len(raw_paths)}."
            )

    LOGGER.info("Loading GT tracking masks from %s.", gt_dir)
    gt_masks = load_stack(gt_paths)
    LOGGER.info("Loading sparse GT segmentation masks from %s.", gt_seg_dir)
    gt_seg_masks_by_frame = load_indexed_stack(gt_seg_paths)
    if not gt_seg_masks_by_frame:
        raise ValueError(f"No GT segmentation masks were found under {gt_seg_dir}.")
    LOGGER.info("Loading ST segmentation masks from %s.", st_seg_dir)
    st_seg_masks_by_frame = load_indexed_stack(st_seg_paths)
    missing_st_frames = sorted(
        frame_index for frame_index in range(len(raw_paths)) if frame_index not in st_seg_masks_by_frame
    )
    if missing_st_frames:
        LOGGER.warning(
            "ST segmentation under %s is missing %s frame(s): %s. "
            "The toy selector will avoid windows that use those frames.",
            st_seg_dir,
            len(missing_st_frames),
            ", ".join(str(frame_index) for frame_index in missing_st_frames),
        )
    lineage_rows = projectio.load_lineage_records(dataset_root, args.track_sequence)
    LOGGER.info("Loading source masks from %s for %s.", extra_seg_root, ", ".join(source_names))
    source_masks = {
        source_name: load_stack(paths, normalize_source=True)
        for source_name, paths in source_paths.items()
    }

    gt_events = extract_gt_events(gt_masks, lineage_rows)
    source_events = extract_source_consensus_events(
        source_masks[source_names[0]],
        source_masks[source_names[1]],
        left_name=source_names[0],
        right_name=source_names[1],
        agreement_iou_threshold=float(args.agreement_iou_threshold),
        overlap_iou_threshold=float(args.overlap_iou_threshold),
    )
    LOGGER.info(
        "Extracted %s GT event(s) and %s source disagreement event(s).",
        len(gt_events),
        len(source_events),
    )

    selection = select_best_crop(
        gt_events + source_events,
        gt_masks,
        window_frames=int(args.window_frames),
        crop_height=int(args.crop_height),
        crop_width=int(args.crop_width),
        spatial_step=int(args.spatial_step),
        temporal_top_k=int(args.temporal_top_k),
        min_bbox_coverage=float(args.min_bbox_coverage),
        annotated_seg_frames=frozenset(gt_seg_masks_by_frame),
        available_st_frames=frozenset(st_seg_masks_by_frame),
    )
    gt_seg_source_frames = tuple(
        sorted(
            frame_index
            for frame_index in gt_seg_masks_by_frame
            if selection.crop.start_frame <= frame_index <= selection.crop.end_frame
        )
    )
    if not gt_seg_source_frames:
        raise ValueError(
            "The selected crop does not overlap any sparse GT SEG annotations. "
            "Try a different window size or adjust the crop-search settings."
        )
    payload = build_report_payload(
        dataset_root=dataset_root,
        extra_seg_root=extra_seg_output_root,
        output_sequence=output_sequence,
        selection=selection,
        source_names=source_names,
        gt_seg_source_frames=gt_seg_source_frames,
        dry_run=bool(args.dry_run),
    )
    LOGGER.info(
        "Selected frames %s..%s with crop top=%s left=%s height=%s width=%s.",
        selection.crop.start_frame,
        selection.crop.end_frame,
        selection.crop.top,
        selection.crop.left,
        selection.crop.height,
        selection.crop.width,
    )
    LOGGER.info("Covered event kinds: %s", ", ".join(sorted(selection.counts_by_kind)))
    if selection.missing_kinds:
        LOGGER.info("Missing event kinds: %s", ", ".join(selection.missing_kinds))

    manifest_json_path = dataset_root / f"{output_sequence}_manifest.json"
    manifest_text_path = dataset_root / f"{output_sequence}_manifest.txt"
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    write_selection_outputs(
        dataset_root=dataset_root,
        extra_seg_output_root=extra_seg_output_root,
        output_sequence=output_sequence,
        crop=selection.crop,
        raw_paths=raw_paths,
        gt_masks=gt_masks,
        gt_seg_masks_by_frame=gt_seg_masks_by_frame,
        st_seg_masks_by_frame=st_seg_masks_by_frame,
        source_masks=source_masks,
        lineage_rows=lineage_rows,
        force=bool(args.force),
    )
    manifest_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_text_path.write_text(format_report_text(payload), encoding="utf-8")
    LOGGER.info("Wrote toy manifest to %s and %s.", manifest_json_path, manifest_text_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
