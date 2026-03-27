from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True, frozen=True)
class TrackingConfig:
    dataset_root: Path
    extra_seg_root: Path | None
    train_sequence: str = "01"
    track_sequence: str = "02"
    mode: str = "single"
    evaluate_only: bool = False
    seg_source: str = "all"
    consensus_sources: tuple[str, ...] = ("embedseg", "stardist")
    agreement_iou_threshold: float = 0.8
    max_distance: float = 50.0
    output_dir: Path | None = None
    consensus_output_dir: Path | None = None
    log_file: Path | None = None
    model_dir: Path | None = None
    force_retrain: bool = False
    force_retrack: bool = False
    segmentation_reward: float = -105.0
    train_iou_threshold: float = 0.3


@dataclass(slots=True, frozen=True)
class SegmentationSource:
    name: str
    sequence: str
    frame_paths: tuple[Path, ...]
    frame_count: int
    shape: tuple[int, int]
    training_capable: bool = True


@dataclass(slots=True)
class FrameObjects:
    source_name: str
    frame_index: int
    label_image: np.ndarray
    raw_label_ids: tuple[int, ...]
    raw_label_to_index: dict[int, int]
    coords: tuple[np.ndarray, ...]
    centroids: tuple[tuple[float, float], ...]
    areas: tuple[int, ...]
    intensity_std: tuple[float, ...]
    border_distance: tuple[float, ...]

    @property
    def object_count(self) -> int:
        return len(self.raw_label_ids)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.label_image.shape)


@dataclass(slots=True, frozen=True)
class LineageRecord:
    track_id: int
    begin: int
    end: int
    parent: int


@dataclass(slots=True, frozen=True)
class TrackingCheckpoint:
    version: int
    dataset_root: str
    track_sequence: str
    seg_source: str
    selected_sources: tuple[str, ...]
    frame_count: int
    frame_shape: tuple[int, int]
    completed_frame: int
    next_track_id: int
    max_distance: float
    segmentation_reward: float
    lineage_state: dict[int, tuple[int, int, int]]


@dataclass(slots=True)
class TrackingResult:
    selected_sources: tuple[str, ...]
    tracklets: tuple[tuple[int, int, float, float], ...]
    lineage_rows: tuple[LineageRecord, ...]
    tracked_masks: np.ndarray
    metrics: dict[str, object] = field(default_factory=dict)
    output_dir: Path | None = None
    mask_paths: tuple[Path, ...] = field(default_factory=tuple)
    lineage_path: Path | None = None
    metrics_json_path: Path | None = None
    metrics_text_path: Path | None = None


@dataclass(slots=True, frozen=True)
class SavedTrackingSolution:
    source_name: str
    output_dir: Path
    tracked_masks: np.ndarray
    lineage_rows: tuple[LineageRecord, ...]
    frames: tuple[FrameObjects, ...]
    checkpoint: TrackingCheckpoint | None = None


@dataclass(slots=True, frozen=True)
class MatchedObjectPair:
    frame_index: int
    track_id_1: int
    track_id_2: int
    iou: float


@dataclass(slots=True, frozen=True)
class CommonTracklet:
    tracklet_id: int
    begin: int
    end: int
    source_names: tuple[str, str]
    source_track_ids: tuple[int, int]


@dataclass(slots=True, frozen=True)
class HypothesisTracklet:
    tracklet_id: int
    begin: int
    end: int
    source_name: str
    source_track_id: int


@dataclass(slots=True, frozen=True)
class ConsensusMetrics:
    agreement_metrics: dict[str, float]
    input_solution_metrics: dict[str, dict[str, object]]
    common_tracklet_count: int
    hypothesis_tracklet_count: int


@dataclass(slots=True, frozen=True)
class VariantEvaluation:
    variant_name: str
    metrics: dict[str, object]
    output_dir: Path
    mask_paths: tuple[Path, ...]
    lineage_path: Path
    metrics_json_path: Path
    metrics_text_path: Path


@dataclass(slots=True)
class ConsensusResult:
    selected_sources: tuple[str, ...]
    lineage_rows: tuple[LineageRecord, ...]
    output_dir: Path
    premerge_metrics_path: Path
    premerge_metrics_text_path: Path
    variant_comparison_path: Path
    variant_comparison_text_path: Path
    variant_evaluations: dict[str, VariantEvaluation]
