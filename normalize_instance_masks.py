from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from dataio import projectio


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FrameNormalizationStats:
    frame_path: Path
    object_count: int
    split_label_count: int
    extra_component_count: int


def format_component_count_summary(disconnected: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> str:
    preview = list(disconnected)[:10]
    summary = ", ".join(f"{label_id} ({component_count} components)" for label_id, component_count in preview)
    remaining = len(disconnected) - len(preview)
    if remaining > 0:
        summary += f", and {remaining} more"
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a folder of instance-segmentation masks, split disconnected components, "
            "assign one unique label id per connected instance, and save the cleaned masks."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing input .tif/.tiff instance masks.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where cleaned masks will be written. Must be different from --input-dir.",
    )
    parser.add_argument(
        "--label-mode",
        default="sequential",
        choices=("sequential", "preserve"),
        help=(
            "How to assign output ids. 'sequential' relabels each frame to 1..N. "
            "'preserve' keeps original ids when possible and only creates new ids for split components."
        ),
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


def relabel_sequentially(label_image: np.ndarray) -> np.ndarray:
    relabeled = np.zeros(label_image.shape, dtype=np.uint32)
    positive_ids = sorted(int(value) for value in np.unique(label_image) if int(value) > 0)
    for next_label_id, label_id in enumerate(positive_ids, start=1):
        relabeled[label_image == label_id] = next_label_id
    if relabeled.size == 0 or int(np.max(relabeled)) <= np.iinfo(np.uint16).max:
        return relabeled.astype(np.uint16, copy=False)
    return relabeled


def normalize_label_image(label_image: np.ndarray, *, label_mode: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    normalized, split_labels = projectio.normalize_source_label_image(label_image)
    if label_mode == "preserve":
        return normalized, split_labels
    if label_mode == "sequential":
        return relabel_sequentially(normalized), split_labels
    raise ValueError(f"Unsupported label mode '{label_mode}'.")


def normalize_mask_file(
    input_path: Path,
    output_path: Path,
    *,
    label_mode: str,
) -> FrameNormalizationStats:
    label_image = np.asarray(tifffile.imread(input_path))
    if label_image.ndim != 2:
        raise ValueError(f"Expected a 2D label image in {input_path}, found shape {label_image.shape}.")

    normalized, split_labels = normalize_label_image(label_image, label_mode=label_mode)
    disconnected = projectio.disconnected_label_components(normalized)
    if disconnected:
        raise ValueError(
            f"Normalization of {input_path} still produced disconnected label ids: "
            f"{format_component_count_summary(disconnected)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, normalized)
    object_count = int(len(np.unique(normalized[normalized > 0])))
    extra_component_count = int(sum(component_count - 1 for _label_id, component_count in split_labels))
    return FrameNormalizationStats(
        frame_path=input_path,
        object_count=object_count,
        split_label_count=len(split_labels),
        extra_component_count=extra_component_count,
    )


def normalize_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    label_mode: str = "sequential",
) -> tuple[FrameNormalizationStats, ...]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must be different to avoid overwriting the raw masks.")

    paths = projectio.sorted_tiff_paths(input_dir)
    if not paths:
        raise ValueError(f"No TIFF masks were found under {input_dir}.")

    LOGGER.info("Normalizing %s mask(s) from %s into %s using label mode '%s'.", len(paths), input_dir, output_dir, label_mode)
    stats: list[FrameNormalizationStats] = []
    for path in paths:
        frame_stats = normalize_mask_file(
            path,
            output_dir / path.name,
            label_mode=label_mode,
        )
        if frame_stats.split_label_count:
            LOGGER.warning(
                "Frame %s: split %s disconnected label id(s), creating %s extra instance(s).",
                path.name,
                frame_stats.split_label_count,
                frame_stats.extra_component_count,
            )
        else:
            LOGGER.debug("Frame %s: no disconnected label ids found.", path.name)
        stats.append(frame_stats)

    total_split_labels = sum(frame_stats.split_label_count for frame_stats in stats)
    total_extra_components = sum(frame_stats.extra_component_count for frame_stats in stats)
    LOGGER.info(
        "Finished writing %s cleaned mask(s). Split %s disconnected label id(s) and created %s extra connected instance(s).",
        len(stats),
        total_split_labels,
        total_extra_components,
    )
    return tuple(stats)


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    normalize_directory(
        Path(args.input_dir),
        Path(args.output_dir),
        label_mode=args.label_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
