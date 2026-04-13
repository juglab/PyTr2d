from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from dataio import projectio
from tracking.consensus import (
    JOINT_GEOMETRY_MODE,
    OPTIMIZED_JOINT_VARIANT,
    load_render_manifest,
    render_variant_names_from_manifest,
)
from tracking.types import SavedTrackingSolution, TrackingConfig


VIEW_ORDER = ("embedseg", "stardist", "joint")
VIEW_COLORS = {
    "embedseg": (0.92, 0.28, 0.25),
    "stardist": (0.22, 0.52, 0.92),
    "joint": (0.16, 0.72, 0.38),
}


@dataclass(slots=True, frozen=True)
class InspectionViewSelection:
    view_name: str
    solution_name: str
    output_dir: Path
    track_ids: tuple[int, ...]
    selected_masks: np.ndarray
    present_by_frame: tuple[bool, ...]


@dataclass(slots=True, frozen=True)
class InspectionSelection:
    dataset_root: Path
    track_sequence: str
    frames: tuple[int, ...]
    raw_frames: np.ndarray
    views: dict[str, InspectionViewSelection]
    crop_bbox: tuple[int, int, int, int] | None


def load_tracklet_selection(
    dataset_root: Path | str,
    track_sequence: str,
    frames: Sequence[int],
    track_ids_by_view: Mapping[str, Sequence[int]],
    consensus_sources: tuple[str, str] = ("embedseg", "stardist"),
    source_output_dirs: Mapping[str, Path | str] | None = None,
    consensus_output_dir: Path | str | None = None,
    joint_variant: str = OPTIMIZED_JOINT_VARIANT,
) -> InspectionSelection:
    dataset_root = Path(dataset_root)
    requested_frames = _normalize_frames(frames)
    if not requested_frames:
        raise ValueError("Expected at least one frame index.")

    normalized_track_ids = _normalize_track_ids_by_view(track_ids_by_view)
    normalized_source_dirs = _normalize_directory_mapping(source_output_dirs)
    if len(consensus_sources) != 2:
        raise ValueError(f"Expected exactly two consensus sources, found {consensus_sources!r}.")

    raw_sequence = projectio.load_raw_sequence(dataset_root, track_sequence)
    _validate_frame_indices(requested_frames, len(raw_sequence))

    source_solutions = {
        source_name: projectio.load_saved_tracking_solution(
            source_name,
            normalized_source_dirs.get(source_name, projectio.resolve_source_output_dir(dataset_root, track_sequence, source_name)),
            raw_sequence,
        )
        for source_name in VIEW_ORDER
        if source_name != "joint"
    }

    consensus_root = (
        Path(consensus_output_dir)
        if consensus_output_dir is not None
        else projectio.resolve_consensus_output_dir(
            TrackingConfig(
                dataset_root=dataset_root,
                extra_seg_root=None,
                track_sequence=track_sequence,
                consensus_sources=tuple(consensus_sources),
            )
        )
    )
    resolved_joint_variant = _resolve_joint_variant(consensus_root, tuple(consensus_sources), joint_variant)
    joint_output_dir = consensus_root / resolved_joint_variant
    joint_solution = projectio.load_saved_tracking_solution(resolved_joint_variant, joint_output_dir, raw_sequence)

    selected_raw_frames = np.stack([np.asarray(raw_sequence[frame_index]) for frame_index in requested_frames])
    views: dict[str, InspectionViewSelection] = {}
    solutions_by_view: dict[str, tuple[SavedTrackingSolution, str, Path]] = {
        "embedseg": (source_solutions["embedseg"], "embedseg", normalized_source_dirs.get("embedseg", projectio.resolve_source_output_dir(dataset_root, track_sequence, "embedseg"))),
        "stardist": (source_solutions["stardist"], "stardist", normalized_source_dirs.get("stardist", projectio.resolve_source_output_dir(dataset_root, track_sequence, "stardist"))),
        "joint": (joint_solution, resolved_joint_variant, joint_output_dir),
    }

    for view_name in VIEW_ORDER:
        solution, solution_name, output_dir = solutions_by_view[view_name]
        selected_masks = _extract_selected_masks(
            solution=solution,
            view_name=view_name,
            requested_frames=requested_frames,
            track_ids=normalized_track_ids[view_name],
        )
        views[view_name] = InspectionViewSelection(
            view_name=view_name,
            solution_name=solution_name,
            output_dir=output_dir,
            track_ids=normalized_track_ids[view_name],
            selected_masks=selected_masks,
            present_by_frame=tuple(bool(np.any(frame_mask > 0)) for frame_mask in selected_masks),
        )

    return InspectionSelection(
        dataset_root=dataset_root,
        track_sequence=track_sequence,
        frames=requested_frames,
        raw_frames=selected_raw_frames,
        views=views,
        crop_bbox=_compute_crop_bbox(tuple(view.selected_masks for view in views.values())),
    )


def plot_tracklet_selection(
    selection: InspectionSelection,
    crop: str | tuple[int, int, int, int] = "tight",
    padding: int = 20,
    overlay_alpha: float = 0.35,
    show_titles: bool = True,
):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environments without matplotlib.
        raise RuntimeError(
            "plot_tracklet_selection requires matplotlib. Install it in the notebook environment to render overlays."
        ) from exc

    if padding < 0:
        raise ValueError(f"Expected non-negative padding, found {padding}.")
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError(f"Expected overlay_alpha in [0, 1], found {overlay_alpha}.")

    row_slice, col_slice = _resolve_plot_slices(crop, selection.crop_bbox, padding, tuple(int(value) for value in selection.raw_frames.shape[1:]))
    figure, axes = plt.subplots(len(selection.frames), len(VIEW_ORDER), squeeze=False)
    axes_grid = np.asarray(axes, dtype=object).reshape(len(selection.frames), len(VIEW_ORDER))

    for row_index, frame_index in enumerate(selection.frames):
        raw_frame = selection.raw_frames[row_index][row_slice, col_slice]
        for column_index, view_name in enumerate(VIEW_ORDER):
            axis = axes_grid[row_index, column_index]
            view = selection.views[view_name]
            mask = view.selected_masks[row_index][row_slice, col_slice]

            axis.imshow(raw_frame, cmap="gray")
            axis.imshow(_rgba_overlay(mask > 0, VIEW_COLORS[view_name], overlay_alpha))
            axis.set_xticks([])
            axis.set_yticks([])

            if row_index == 0 and show_titles:
                axis.set_title(_panel_title(view))
            if column_index == 0:
                axis.set_ylabel(f"frame {frame_index}")
            if not view.track_ids:
                axis.text(0.5, 0.08, "no ids selected", ha="center", va="bottom", transform=axis.transAxes)
            elif not view.present_by_frame[row_index]:
                axis.text(0.5, 0.08, "not present", ha="center", va="bottom", transform=axis.transAxes)

    figure.tight_layout()
    return figure


def _normalize_frames(frames: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(frame_index) for frame_index in frames)


def _normalize_track_ids_by_view(track_ids_by_view: Mapping[str, Sequence[int]]) -> dict[str, tuple[int, ...]]:
    unknown_views = sorted(set(track_ids_by_view) - set(VIEW_ORDER))
    if unknown_views:
        raise ValueError(f"Unknown track-id view(s): {', '.join(unknown_views)}.")
    return {
        view_name: _unique_ints(track_ids_by_view.get(view_name, ()))
        for view_name in VIEW_ORDER
    }


def _normalize_directory_mapping(directory_mapping: Mapping[str, Path | str] | None) -> dict[str, Path]:
    if directory_mapping is None:
        return {}
    unknown_views = sorted(set(directory_mapping) - {"embedseg", "stardist"})
    if unknown_views:
        raise ValueError(f"Unknown source output override(s): {', '.join(unknown_views)}.")
    return {view_name: Path(path) for view_name, path in directory_mapping.items()}


def _unique_ints(values: Sequence[int]) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        normalized = int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _validate_frame_indices(frames: tuple[int, ...], frame_count: int) -> None:
    invalid = [frame_index for frame_index in frames if frame_index < 0 or frame_index >= frame_count]
    if invalid:
        raise ValueError(
            f"Requested frame indices {invalid} are out of range for a sequence with {frame_count} frame(s)."
        )


def _resolve_joint_variant(
    consensus_output_dir: Path,
    consensus_sources: tuple[str, str],
    joint_variant: str,
) -> str:
    manifest = load_render_manifest(consensus_output_dir)
    candidates: list[str] = []
    if manifest is not None:
        candidates.extend(render_variant_names_from_manifest(manifest, consensus_sources, JOINT_GEOMETRY_MODE))
    candidates.append(joint_variant)
    if OPTIMIZED_JOINT_VARIANT not in candidates:
        candidates.append(OPTIMIZED_JOINT_VARIANT)

    ordered_candidates = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if (consensus_output_dir / candidate).exists():
            return candidate
    return ordered_candidates[0]


def _extract_selected_masks(
    solution: SavedTrackingSolution,
    view_name: str,
    requested_frames: tuple[int, ...],
    track_ids: tuple[int, ...],
) -> np.ndarray:
    selected_masks = np.zeros((len(requested_frames), *solution.tracked_masks.shape[1:]), dtype=np.uint16)
    if not track_ids:
        return selected_masks

    missing_ids: list[int] = []
    for track_id in track_ids:
        seen = False
        for row_index, frame_index in enumerate(requested_frames):
            frame_mask = solution.tracked_masks[frame_index]
            selected = frame_mask == track_id
            if not np.any(selected):
                continue
            seen = True
            selected_masks[row_index][selected] = np.uint16(track_id)
        if not seen:
            missing_ids.append(track_id)

    if missing_ids:
        raise ValueError(
            f"Requested track ids {missing_ids} were not found in view '{view_name}' across frames {list(requested_frames)}."
        )
    return selected_masks


def _compute_crop_bbox(selected_masks_by_view: tuple[np.ndarray, ...]) -> tuple[int, int, int, int] | None:
    if not selected_masks_by_view:
        return None
    union_mask = np.zeros(selected_masks_by_view[0].shape[1:], dtype=bool)
    for selected_masks in selected_masks_by_view:
        union_mask |= np.any(selected_masks > 0, axis=0)
    if not np.any(union_mask):
        return None
    rows, cols = np.nonzero(union_mask)
    return (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)


def _resolve_plot_slices(
    crop: str | tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int] | None,
    padding: int,
    frame_shape: tuple[int, int],
) -> tuple[slice, slice]:
    if crop == "tight":
        bbox = _expand_bbox(crop_bbox, padding, frame_shape) if crop_bbox is not None else (0, 0, *frame_shape)
    elif crop == "full":
        bbox = (0, 0, *frame_shape)
    elif isinstance(crop, tuple) and len(crop) == 4:
        bbox = _clip_bbox(tuple(int(value) for value in crop), frame_shape)
    else:
        raise ValueError("crop must be 'tight', 'full', or a 4-tuple bbox.")
    return slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    padding: int,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    top, left, bottom, right = bbox
    return _clip_bbox((top - padding, left - padding, bottom + padding, right + padding), frame_shape)


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    top, left, bottom, right = bbox
    height, width = frame_shape
    clipped = (
        max(0, min(int(top), height)),
        max(0, min(int(left), width)),
        max(0, min(int(bottom), height)),
        max(0, min(int(right), width)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Invalid crop bbox {bbox!r} for frame shape {frame_shape}.")
    return clipped


def _rgba_overlay(mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def _panel_title(view: InspectionViewSelection) -> str:
    label = "none" if not view.track_ids else ", ".join(str(track_id) for track_id in view.track_ids)
    if view.view_name == "joint":
        return f"joint\nids: {label}"
    return f"{view.view_name}\nids: {label}"
