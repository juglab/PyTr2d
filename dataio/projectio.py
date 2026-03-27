from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

from tracking.types import (
    FrameObjects,
    LineageRecord,
    SavedTrackingSolution,
    SegmentationSource,
    TrackingCheckpoint,
    TrackingConfig,
)


TIFF_SUFFIXES = {".tif", ".tiff"}
CHECKPOINT_FILENAME = "tracking_checkpoint.json"
LOGGER = logging.getLogger(__name__)


def extract_frame_index(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    if not digits:
        raise ValueError(f"Could not extract a frame index from {path}")
    return int(digits[-1])


def sorted_tiff_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES]
    return tuple(sorted(paths, key=lambda path: (extract_frame_index(path), path.name)))


def default_extra_seg_root(dataset_root: Path) -> Path | None:
    candidate = dataset_root.parent / "Segmentations"
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def resolve_output_dir(config: TrackingConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    return Path("outputs") / config.dataset_root.name / config.track_sequence / config.seg_source


def resolve_source_output_dir(dataset_root: Path, track_sequence: str, source_name: str) -> Path:
    return Path("outputs") / dataset_root.name / track_sequence / source_name


def resolve_consensus_output_dir(config: TrackingConfig) -> Path:
    if config.consensus_output_dir is not None:
        return config.consensus_output_dir
    source_tag = "_".join(config.consensus_sources)
    return Path("outputs") / config.dataset_root.name / config.track_sequence / f"consensus_{source_tag}"


def discover_segmentation_sources(
    dataset_root: Path,
    sequence: str,
    extra_seg_root: Path | None = None,
    include_external: bool = False,
) -> dict[str, SegmentationSource]:
    LOGGER.info(
        "Discovering segmentation sources for sequence %s (include external: %s).",
        sequence,
        include_external,
    )
    sources: dict[str, SegmentationSource] = {}

    builtins = {
        "st": dataset_root / f"{sequence}_ST" / "SEG",
        "err_seg": dataset_root / f"{sequence}_ERR_SEG",
    }
    for source_name, source_dir in builtins.items():
        source = _build_source(source_name, sequence, source_dir, training_capable=True)
        if source is not None:
            sources[source_name] = source

    if include_external and extra_seg_root is not None and extra_seg_root.exists():
        for child in sorted(extra_seg_root.iterdir()):
            if not child.is_dir():
                continue
            source = _build_source(child.name, sequence, child, training_capable=False)
            if source is None:
                continue
            if source.name in sources:
                raise ValueError(f"Duplicate segmentation source name discovered: {source.name}")
            sources[source.name] = source

    LOGGER.info(
        "Discovered %s segmentation source(s): %s",
        len(sources),
        ", ".join(sorted(sources)) if sources else "<none>",
    )
    return sources


def select_segmentation_sources(
    available_sources: dict[str, SegmentationSource],
    seg_source: str,
) -> dict[str, SegmentationSource]:
    if not available_sources:
        raise ValueError("No segmentation sources were discovered.")
    if seg_source == "all":
        return dict(sorted(available_sources.items()))
    if seg_source not in available_sources:
        options = ", ".join(sorted(available_sources))
        raise ValueError(f"Unknown segmentation source '{seg_source}'. Available sources: {options}")
    return {seg_source: available_sources[seg_source]}


def load_raw_sequence(dataset_root: Path, sequence: str) -> np.ndarray:
    sequence_dir = dataset_root / sequence
    paths = sorted_tiff_paths(sequence_dir)
    if not paths:
        raise ValueError(f"No raw frames found under {sequence_dir}")
    LOGGER.info("Loading raw sequence %s from %s.", sequence, sequence_dir)
    stack = _load_image_stack(paths)
    LOGGER.info(
        "Loaded raw sequence %s with %s frame(s), shape %s, dtype %s.",
        sequence,
        len(stack),
        tuple(stack.shape[1:]),
        stack.dtype,
    )
    return stack


def load_source_frame_objects(source: SegmentationSource, raw_frames: np.ndarray) -> list[FrameObjects]:
    if source.frame_count != len(raw_frames):
        raise ValueError(
            f"Segmentation source '{source.name}' has {source.frame_count} frames, "
            f"but the raw sequence has {len(raw_frames)} frames."
        )

    LOGGER.info(
        "Loading segmentation source '%s' for sequence %s from %s frame(s).",
        source.name,
        source.sequence,
        source.frame_count,
    )
    frames: list[FrameObjects] = []
    expected_shape = tuple(int(value) for value in raw_frames[0].shape)
    for frame_index, (frame_path, raw_frame) in enumerate(zip(source.frame_paths, raw_frames, strict=True)):
        label_image = np.asarray(tifffile.imread(frame_path))
        _validate_2d_image(label_image, frame_path)
        if tuple(label_image.shape) != expected_shape:
            raise ValueError(
                f"Segmentation frame {frame_path} has shape {label_image.shape}, "
                f"expected {expected_shape}."
            )
        frame_objects = build_frame_objects(source.name, frame_index, label_image, raw_frame)
        LOGGER.debug(
            "Loaded %s frame %03d from %s with %s object(s).",
            source.name,
            frame_index,
            frame_path.name,
            frame_objects.object_count,
        )
        frames.append(frame_objects)
    object_counts = [frame.object_count for frame in frames]
    LOGGER.info(
        "Loaded source '%s': %s frame(s), object count min/avg/max = %s/%.1f/%s.",
        source.name,
        len(frames),
        min(object_counts) if object_counts else 0,
        float(np.mean(object_counts)) if object_counts else 0.0,
        max(object_counts) if object_counts else 0,
    )
    return frames


def load_gt_frame_objects(dataset_root: Path, sequence: str, raw_frames: np.ndarray) -> list[FrameObjects]:
    tracking_dir = dataset_root / f"{sequence}_GT" / "TRA"
    paths = sorted_tiff_paths(tracking_dir)
    if not paths:
        raise ValueError(f"No tracking GT masks found under {tracking_dir}")
    if len(paths) != len(raw_frames):
        raise ValueError(
            f"Tracking GT under {tracking_dir} has {len(paths)} frames, "
            f"but the raw sequence has {len(raw_frames)} frames."
        )

    LOGGER.info("Loading GT tracking masks for sequence %s from %s.", sequence, tracking_dir)
    frames: list[FrameObjects] = []
    expected_shape = tuple(int(value) for value in raw_frames[0].shape)
    for frame_index, (frame_path, raw_frame) in enumerate(zip(paths, raw_frames, strict=True)):
        label_image = np.asarray(tifffile.imread(frame_path))
        _validate_2d_image(label_image, frame_path)
        if tuple(label_image.shape) != expected_shape:
            raise ValueError(
                f"Tracking GT frame {frame_path} has shape {label_image.shape}, expected {expected_shape}."
            )
        frame_objects = build_frame_objects("gt", frame_index, label_image, raw_frame)
        LOGGER.debug(
            "Loaded GT frame %03d from %s with %s object(s).",
            frame_index,
            frame_path.name,
            frame_objects.object_count,
        )
        frames.append(frame_objects)
    LOGGER.info("Loaded %s GT frame(s) for sequence %s.", len(frames), sequence)
    return frames


def load_lineage_records(dataset_root: Path, sequence: str) -> dict[int, LineageRecord]:
    lineage_path = dataset_root / f"{sequence}_GT" / "TRA" / "man_track.txt"
    if not lineage_path.exists():
        raise ValueError(f"Missing lineage file: {lineage_path}")

    LOGGER.info("Loading lineage records from %s.", lineage_path)
    records: dict[int, LineageRecord] = {}
    with lineage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            track_id, begin, end, parent = (int(value) for value in stripped.split())
            records[track_id] = LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent)
    LOGGER.info("Loaded %s lineage record(s).", len(records))
    return records


def build_frame_objects(
    source_name: str,
    frame_index: int,
    label_image: np.ndarray,
    raw_frame: np.ndarray,
) -> FrameObjects:
    props = regionprops(label_image, raw_frame, extra_properties=[_intensity_std])
    height, width = label_image.shape

    raw_label_ids = tuple(int(prop.label) for prop in props)
    coords = tuple(np.asarray(prop.coords, dtype=np.int32) for prop in props)
    centroids = tuple((float(prop.centroid[0]), float(prop.centroid[1])) for prop in props)
    areas = tuple(int(prop.area) for prop in props)
    intensity_std = tuple(float(prop.intensity_std) for prop in props)
    border_distance = tuple(
        float(min(row, height - 1 - row, col, width - 1 - col))
        for row, col in centroids
    )

    return FrameObjects(
        source_name=source_name,
        frame_index=frame_index,
        label_image=np.asarray(label_image),
        raw_label_ids=raw_label_ids,
        raw_label_to_index={label_id: index for index, label_id in enumerate(raw_label_ids)},
        coords=coords,
        centroids=centroids,
        areas=areas,
        intensity_std=intensity_std,
        border_distance=border_distance,
    )


def write_tracking_outputs(
    output_dir: Path,
    tracked_masks: np.ndarray,
    lineage_rows: tuple[LineageRecord, ...],
) -> tuple[tuple[Path, ...], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing tracking outputs to %s.", output_dir)

    mask_paths: list[Path] = []
    for frame_index, mask in enumerate(tracked_masks):
        mask_path = write_tracking_mask(output_dir, frame_index, mask)
        mask_paths.append(mask_path)

    lineage_path = write_lineage_rows(output_dir, lineage_rows)

    LOGGER.info(
        "Finished writing %s mask(s) and lineage file %s.",
        len(mask_paths),
        lineage_path.name,
    )
    return tuple(mask_paths), lineage_path


def write_tracking_mask(output_dir: Path, frame_index: int, mask: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"mask{frame_index:03d}.tif"
    tifffile.imwrite(mask_path, np.asarray(mask, dtype=np.uint16))
    LOGGER.debug("Wrote tracked mask %s.", mask_path.name)
    return mask_path


def load_tracking_mask(output_dir: Path, frame_index: int) -> np.ndarray:
    mask_path = output_dir / f"mask{frame_index:03d}.tif"
    if not mask_path.exists():
        raise ValueError(f"Missing tracked mask for frame {frame_index}: {mask_path}")
    image = np.asarray(tifffile.imread(mask_path))
    _validate_2d_image(image, mask_path)
    return image


def load_lineage_rows_from_output(output_dir: Path) -> tuple[LineageRecord, ...]:
    lineage_path = output_dir / "res_track.txt"
    if not lineage_path.exists():
        raise ValueError(f"Missing lineage file: {lineage_path}")

    rows: list[LineageRecord] = []
    with lineage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            track_id, begin, end, parent = (int(value) for value in stripped.split())
            rows.append(LineageRecord(track_id=track_id, begin=begin, end=end, parent=parent))
    return tuple(rows)


def is_saved_tracking_complete(output_dir: Path, frame_count: int) -> bool:
    checkpoint = load_tracking_checkpoint(output_dir)
    if checkpoint is not None and checkpoint.completed_frame != frame_count - 1:
        return False
    if not (output_dir / "res_track.txt").exists():
        return False
    return all((output_dir / f"mask{frame_index:03d}.tif").exists() for frame_index in range(frame_count))


def load_saved_tracking_solution(
    source_name: str,
    output_dir: Path,
    raw_frames: np.ndarray,
) -> SavedTrackingSolution:
    frame_count = len(raw_frames)
    if not is_saved_tracking_complete(output_dir, frame_count):
        raise ValueError(f"Saved tracking output under {output_dir} is missing files or incomplete.")

    checkpoint = load_tracking_checkpoint(output_dir)
    masks = np.zeros((frame_count, *raw_frames[0].shape), dtype=np.uint16)
    frames: list[FrameObjects] = []
    for frame_index in range(frame_count):
        mask = np.asarray(load_tracking_mask(output_dir, frame_index), dtype=np.uint16)
        if tuple(mask.shape) != tuple(int(value) for value in raw_frames[frame_index].shape):
            raise ValueError(
                f"Saved mask {output_dir / f'mask{frame_index:03d}.tif'} has shape {mask.shape}, "
                f"expected {tuple(raw_frames[frame_index].shape)}."
            )
        masks[frame_index] = mask
        frames.append(build_frame_objects(source_name, frame_index, mask, raw_frames[frame_index]))

    solution = SavedTrackingSolution(
        source_name=source_name,
        output_dir=output_dir,
        tracked_masks=masks,
        lineage_rows=load_lineage_rows_from_output(output_dir),
        frames=tuple(frames),
        checkpoint=checkpoint,
    )
    LOGGER.info(
        "Loaded saved tracking solution '%s' from %s with %s frame(s) and %s lineage row(s).",
        source_name,
        output_dir,
        frame_count,
        len(solution.lineage_rows),
    )
    return solution


def write_lineage_rows(output_dir: Path, lineage_rows: tuple[LineageRecord, ...]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = output_dir / "res_track.txt"
    with lineage_path.open("w", encoding="utf-8") as handle:
        for row in lineage_rows:
            handle.write(f"{row.track_id} {row.begin} {row.end} {row.parent}\n")
    LOGGER.debug("Wrote lineage file %s.", lineage_path.name)
    return lineage_path


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_FILENAME


def write_tracking_checkpoint(output_dir: Path, checkpoint: TrackingCheckpoint) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(output_dir)
    payload = {
        "version": checkpoint.version,
        "dataset_root": checkpoint.dataset_root,
        "track_sequence": checkpoint.track_sequence,
        "seg_source": checkpoint.seg_source,
        "selected_sources": list(checkpoint.selected_sources),
        "frame_count": checkpoint.frame_count,
        "frame_shape": list(checkpoint.frame_shape),
        "completed_frame": checkpoint.completed_frame,
        "next_track_id": checkpoint.next_track_id,
        "max_distance": checkpoint.max_distance,
        "segmentation_reward": checkpoint.segmentation_reward,
        "lineage_state": {
            str(track_id): list(values)
            for track_id, values in checkpoint.lineage_state.items()
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    LOGGER.debug("Wrote tracking checkpoint %s.", path.name)
    return path


def load_tracking_checkpoint(output_dir: Path) -> TrackingCheckpoint | None:
    path = checkpoint_path(output_dir)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return TrackingCheckpoint(
        version=int(payload["version"]),
        dataset_root=str(payload["dataset_root"]),
        track_sequence=str(payload["track_sequence"]),
        seg_source=str(payload["seg_source"]),
        selected_sources=tuple(str(value) for value in payload["selected_sources"]),
        frame_count=int(payload["frame_count"]),
        frame_shape=tuple(int(value) for value in payload["frame_shape"]),
        completed_frame=int(payload["completed_frame"]),
        next_track_id=int(payload["next_track_id"]),
        max_distance=float(payload["max_distance"]),
        segmentation_reward=float(payload["segmentation_reward"]),
        lineage_state={
            int(track_id): tuple(int(value) for value in values)
            for track_id, values in payload["lineage_state"].items()
        },
    )


def _build_source(
    name: str,
    sequence: str,
    directory: Path,
    training_capable: bool,
) -> SegmentationSource | None:
    paths = sorted_tiff_paths(directory)
    if not paths:
        LOGGER.debug("Skipping empty segmentation source directory %s.", directory)
        return None

    first_image = np.asarray(tifffile.imread(paths[0]))
    _validate_2d_image(first_image, paths[0])
    source = SegmentationSource(
        name=name,
        sequence=sequence,
        frame_paths=paths,
        frame_count=len(paths),
        shape=tuple(int(value) for value in first_image.shape),
        training_capable=training_capable,
    )
    LOGGER.info(
        "Registered source '%s' from %s with %s frame(s), shape %s, training capable: %s.",
        source.name,
        directory,
        source.frame_count,
        source.shape,
        source.training_capable,
    )
    return source


def _load_image_stack(paths: tuple[Path, ...]) -> np.ndarray:
    stack: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for path in paths:
        image = np.asarray(tifffile.imread(path))
        _validate_2d_image(image, path)
        shape = tuple(int(value) for value in image.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Inconsistent image shapes in {path.parent}: saw {shape}, expected {expected_shape}.")
        stack.append(image)
    return np.stack(stack)


def _validate_2d_image(image: np.ndarray, path: Path) -> None:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image at {path}, found shape {image.shape}.")


def _intensity_std(region: np.ndarray, intensities: np.ndarray) -> float:
    return float(np.std(intensities[region]))
