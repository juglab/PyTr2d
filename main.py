from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

from dataio import projectio
from tracking.consensus import evaluate_saved_consensus_outputs, solve_consensus_tracking
from tracking.ctc_evaluation import evaluate_result_with_ctc, log_ctc_evaluation
from tracking.random_forest import (
    EventScorers,
    RandomForestEventTrainer,
    load_event_scorers,
    resolve_model_path,
    save_event_scorers,
)
from tracking.reporting import format_metrics_report, write_json, write_text
from tracking.trackingsolver import solve_tracking
from tracking.types import ConsensusResult, SavedTrackingSolution, TrackingConfig, TrackingResult


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train on sequence 01 and track sequence 02 with one or more segmentation sources.")
    parser.add_argument(
        "--mode",
        default="single",
        choices=("single", "consensus"),
        help="Run either the original single-source tracker or the new consensus merge mode.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip tracking and recompute metrics only from saved outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        default="./data/Fluo-N2DL-HeLa_train/Fluo-N2DL-HeLa",
        help="Path to the dataset root containing raw, ST, ERR, and GT sequence folders.",
    )
    parser.add_argument(
        "--extra-seg-root",
        default=None,
        help="Optional path to extra segmentation folders such as embedseg or stardist.",
    )
    parser.add_argument("--train-sequence", default="01", help="Sequence used to train the event scorers.")
    parser.add_argument("--track-sequence", default="02", help="Sequence used for tracking.")
    parser.add_argument(
        "--seg-source",
        default="all",
        help="Segmentation source to track, or 'all' to use every discovered source.",
    )
    parser.add_argument(
        "--consensus-sources",
        nargs=2,
        default=("embedseg", "stardist"),
        metavar=("SOURCE_1", "SOURCE_2"),
        help="The two external source results to merge in consensus mode.",
    )
    parser.add_argument(
        "--agreement-iou-threshold",
        type=float,
        default=0.8,
        help="Minimum IoU required to consider two tracked objects common between the two source solutions.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Maximum centroid distance for move and division candidates.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for tracked masks and res_track.txt.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Defaults to <output-dir>/run.log.",
    )
    parser.add_argument(
        "--consensus-output-dir",
        default=None,
        help="Optional root directory for consensus outputs. Defaults to outputs/<dataset>/<track-sequence>/consensus_<sources>.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional root directory where saved event-classifier bundles are stored.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any saved classifier bundle and retrain the event scorers.",
    )
    parser.add_argument(
        "--force-retrack",
        action="store_true",
        help="Ignore any saved tracking checkpoint and restart tracking from frame 0.",
    )
    parser.add_argument(
        "--segmentation-reward",
        type=float,
        default=-105.0,
        help="Constant activation reward assigned to each active segmentation hypothesis.",
    )
    parser.add_argument(
        "--train-iou-threshold",
        type=float,
        default=0.3,
        help="Minimum IoU required to match a training segmentation object to a GT object.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Terminal logging verbosity.",
    )
    return parser


def args_to_config(args: argparse.Namespace) -> TrackingConfig:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    extra_seg_root = (
        Path(args.extra_seg_root).expanduser().resolve()
        if args.extra_seg_root
        else projectio.default_extra_seg_root(dataset_root)
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    consensus_output_dir = Path(args.consensus_output_dir).expanduser().resolve() if args.consensus_output_dir else None
    if args.mode == "consensus" and consensus_output_dir is None and output_dir is not None:
        consensus_output_dir = output_dir
        output_dir = None
    if args.mode == "consensus":
        default_log_root = consensus_output_dir if consensus_output_dir is not None else (
            Path("outputs") / dataset_root.name / args.track_sequence / f"consensus_{'_'.join(args.consensus_sources)}"
        )
    else:
        default_log_root = output_dir if output_dir is not None else Path("outputs") / dataset_root.name / args.track_sequence / args.seg_source
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else default_log_root / "run.log"
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None
    return TrackingConfig(
        dataset_root=dataset_root,
        extra_seg_root=extra_seg_root,
        train_sequence=args.train_sequence,
        track_sequence=args.track_sequence,
        mode=args.mode,
        evaluate_only=args.evaluate_only,
        seg_source=args.seg_source,
        consensus_sources=tuple(args.consensus_sources),
        agreement_iou_threshold=args.agreement_iou_threshold,
        max_distance=args.max_distance,
        output_dir=output_dir,
        consensus_output_dir=consensus_output_dir,
        log_file=log_file,
        model_dir=model_dir,
        force_retrain=args.force_retrain,
        force_retrack=args.force_retrack,
        segmentation_reward=args.segmentation_reward,
        train_iou_threshold=args.train_iou_threshold,
    )


def configure_logging(level_name: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def load_or_train_event_scorers(
    config: TrackingConfig,
    trainer: RandomForestEventTrainer,
    fit_scorers: Callable[[], EventScorers],
) -> EventScorers:
    model_path = resolve_model_path(config)
    expected_metadata = trainer.bundle_metadata(config.dataset_root.name, config.train_sequence)
    LOGGER.info("Classifier bundle path: %s", model_path)

    if config.force_retrain:
        LOGGER.info("Ignoring saved classifiers because --force-retrain was requested.")
    else:
        if model_path.exists():
            LOGGER.info("Loading saved event classifiers.")
            try:
                scorers = load_event_scorers(model_path, expected_metadata)
            except ValueError as exc:
                LOGGER.info("Saved classifiers are incompatible and will be rebuilt: %s", exc)
            else:
                LOGGER.info("Loaded saved event classifiers from %s.", model_path)
                return scorers
        else:
            LOGGER.info("No saved event classifiers found; training a new bundle.")

    scorers = fit_scorers()
    LOGGER.info("Saving event classifiers to %s.", model_path)
    save_event_scorers(model_path, scorers, expected_metadata)
    return scorers


def run_tracking(config: TrackingConfig) -> TrackingResult | ConsensusResult:
    if not config.dataset_root.exists():
        raise ValueError(f"Dataset root does not exist: {config.dataset_root}")

    LOGGER.info("Starting tracking run.")
    LOGGER.info("Dataset root: %s", config.dataset_root)
    LOGGER.info("Run mode: %s", config.mode)
    LOGGER.info("Evaluate only: %s", config.evaluate_only)
    LOGGER.info("Training sequence: %s | Tracking sequence: %s", config.train_sequence, config.track_sequence)
    if config.mode == "single":
        LOGGER.info("Segmentation source selection: %s", config.seg_source)
        LOGGER.info("Output directory: %s", projectio.resolve_output_dir(config))
    else:
        LOGGER.info("Consensus source selection: %s", ", ".join(config.consensus_sources))
        LOGGER.info("Consensus output directory: %s", projectio.resolve_consensus_output_dir(config))
    LOGGER.info("Extra segmentation root: %s", config.extra_seg_root if config.extra_seg_root else "<none>")
    LOGGER.info("Model directory root: %s", config.model_dir if config.model_dir else Path("models"))
    LOGGER.info("Log file: %s", config.log_file)
    LOGGER.info("Force retrack: %s", config.force_retrack)

    if config.evaluate_only:
        if config.mode == "consensus":
            return _evaluate_consensus_tracking(config)
        return _evaluate_single_tracking(config)

    trainer = RandomForestEventTrainer(
        max_distance=config.max_distance,
        train_iou_threshold=config.train_iou_threshold,
    )
    scorers = load_or_train_event_scorers(
        config=config,
        trainer=trainer,
        fit_scorers=lambda: _fit_event_scorers(config, trainer),
    )

    if config.mode == "consensus":
        return _run_consensus_tracking(config, scorers)
    return _run_single_tracking(config, scorers)


def _run_single_tracking(
    config: TrackingConfig,
    scorers: EventScorers,
) -> TrackingResult:
    LOGGER.info("Loading raw tracking frames.")
    track_raw_frames = projectio.load_raw_sequence(config.dataset_root, config.track_sequence)
    LOGGER.info("Discovering tracking segmentation sources.")
    discovered_track_sources = projectio.discover_segmentation_sources(
        config.dataset_root,
        config.track_sequence,
        config.extra_seg_root,
        include_external=True,
    )
    selected_track_sources = projectio.select_segmentation_sources(discovered_track_sources, config.seg_source)
    LOGGER.info("Selected tracking sources: %s", ", ".join(sorted(selected_track_sources)))
    LOGGER.info("Loading tracking segmentation objects.")
    track_frames_by_source = {
        name: projectio.load_source_frame_objects(source, track_raw_frames)
        for name, source in selected_track_sources.items()
    }

    LOGGER.info("Running consecutive-frame tracking ILPs.")
    result = solve_tracking(config, track_raw_frames, track_frames_by_source, scorers)
    output_dir = projectio.resolve_output_dir(config)
    LOGGER.info("Writing tracked masks and lineage output.")
    mask_paths, lineage_path = projectio.write_tracking_outputs(output_dir, result.tracked_masks, result.lineage_rows)
    finalize_single_tracking_metrics(config, result, output_dir, mask_paths=mask_paths, lineage_path=lineage_path)
    LOGGER.info(
        "Tracking run completed: %s track rows, %s output masks.",
        len(result.lineage_rows),
        len(result.mask_paths),
    )
    return result


def _evaluate_single_tracking(config: TrackingConfig) -> TrackingResult:
    LOGGER.info("Running evaluation-only mode for saved single-source outputs.")
    raw_frames = projectio.load_raw_sequence(config.dataset_root, config.track_sequence)
    output_dir = projectio.resolve_output_dir(config)
    result = projectio.load_saved_tracking_solution(config.seg_source, output_dir, raw_frames)
    tracking_result = TrackingResult(
        selected_sources=(config.seg_source,),
        tracklets=(),
        lineage_rows=result.lineage_rows,
        tracked_masks=result.tracked_masks,
    )
    finalize_single_tracking_metrics(
        config,
        tracking_result,
        output_dir,
        mask_paths=tuple(output_dir / f"mask{frame_index:03d}.tif" for frame_index in range(len(result.tracked_masks))),
        lineage_path=output_dir / "res_track.txt",
    )
    LOGGER.info("Finished evaluation-only mode for %s.", output_dir)
    return tracking_result


def _run_consensus_tracking(
    config: TrackingConfig,
    scorers: EventScorers,
) -> ConsensusResult:
    forbidden_sources = {"st", "err_seg", "gt"}
    invalid_sources = sorted(source for source in config.consensus_sources if source in forbidden_sources)
    if invalid_sources:
        raise ValueError(f"Consensus mode only supports external source results. Invalid sources: {', '.join(invalid_sources)}")

    LOGGER.info("Loading raw tracking frames for consensus mode.")
    raw_frames = projectio.load_raw_sequence(config.dataset_root, config.track_sequence)
    frame_count = len(raw_frames)
    solutions_by_source: dict[str, SavedTrackingSolution] = {}
    for source_name in config.consensus_sources:
        source_output_dir = projectio.resolve_source_output_dir(config.dataset_root, config.track_sequence, source_name)
        needs_run = config.force_retrack or not projectio.is_saved_tracking_complete(source_output_dir, frame_count)
        if needs_run:
            LOGGER.info(
                "Consensus input '%s' is missing or will be rebuilt; running single-source tracking into %s.",
                source_name,
                source_output_dir,
            )
            single_config = TrackingConfig(
                dataset_root=config.dataset_root,
                extra_seg_root=config.extra_seg_root,
                train_sequence=config.train_sequence,
                track_sequence=config.track_sequence,
                mode="single",
                seg_source=source_name,
                consensus_sources=config.consensus_sources,
                agreement_iou_threshold=config.agreement_iou_threshold,
                max_distance=config.max_distance,
                output_dir=source_output_dir,
                consensus_output_dir=config.consensus_output_dir,
                log_file=config.log_file,
                model_dir=config.model_dir,
                force_retrain=False,
                force_retrack=config.force_retrack,
                segmentation_reward=config.segmentation_reward,
                train_iou_threshold=config.train_iou_threshold,
            )
            _run_single_tracking(single_config, scorers)
        else:
            LOGGER.info("Using existing saved tracking result for consensus input '%s' from %s.", source_name, source_output_dir)
        solutions_by_source[source_name] = projectio.load_saved_tracking_solution(source_name, source_output_dir, raw_frames)

    LOGGER.info("Running consensus/global tracklet ILP.")
    return solve_consensus_tracking(config, raw_frames, solutions_by_source, scorers)


def _evaluate_consensus_tracking(config: TrackingConfig) -> ConsensusResult:
    LOGGER.info("Running evaluation-only mode for saved consensus outputs.")
    forbidden_sources = {"st", "err_seg", "gt"}
    invalid_sources = sorted(source for source in config.consensus_sources if source in forbidden_sources)
    if invalid_sources:
        raise ValueError(f"Consensus mode only supports external source results. Invalid sources: {', '.join(invalid_sources)}")

    raw_frames = projectio.load_raw_sequence(config.dataset_root, config.track_sequence)
    frame_count = len(raw_frames)
    solutions_by_source: dict[str, SavedTrackingSolution] = {}
    for source_name in config.consensus_sources:
        source_output_dir = projectio.resolve_source_output_dir(config.dataset_root, config.track_sequence, source_name)
        if not projectio.is_saved_tracking_complete(source_output_dir, frame_count):
            raise ValueError(
                f"Saved consensus input '{source_name}' is incomplete under {source_output_dir}. "
                "Run the tracker first without --evaluate-only."
            )
        solutions_by_source[source_name] = projectio.load_saved_tracking_solution(source_name, source_output_dir, raw_frames)

    result = evaluate_saved_consensus_outputs(config, raw_frames, solutions_by_source)
    LOGGER.info("Finished evaluation-only mode for consensus outputs in %s.", result.output_dir)
    return result


def _fit_event_scorers(
    config: TrackingConfig,
    trainer: RandomForestEventTrainer,
) -> EventScorers:
    LOGGER.info("Loading raw training frames.")
    train_raw_frames = projectio.load_raw_sequence(config.dataset_root, config.train_sequence)
    LOGGER.info("Discovering training segmentation sources.")
    training_sources = {
        name: source
        for name, source in projectio.discover_segmentation_sources(
            config.dataset_root,
            config.train_sequence,
            config.extra_seg_root,
            include_external=False,
        ).items()
        if source.training_capable
    }
    if not training_sources:
        raise ValueError("No training-capable segmentation sources were found for the training sequence.")
    LOGGER.info("Training sources: %s", ", ".join(sorted(training_sources)))

    LOGGER.info("Loading training segmentation objects.")
    training_frames_by_source = {
        name: projectio.load_source_frame_objects(source, train_raw_frames)
        for name, source in training_sources.items()
    }
    LOGGER.info("Loading training GT tracking masks and lineage.")
    gt_frames = projectio.load_gt_frame_objects(config.dataset_root, config.train_sequence, train_raw_frames)
    lineage_records = projectio.load_lineage_records(config.dataset_root, config.train_sequence)

    LOGGER.info("Building and fitting event scorers.")
    return trainer.fit(training_frames_by_source, gt_frames, lineage_records)


def summarize_tracking_result(result: TrackingResult) -> dict[str, float]:
    object_counts = [int(len(set(frame[frame > 0].tolist()))) for frame in result.tracked_masks]
    return {
        "track_count": float(len(result.lineage_rows)),
        "division_count": float(sum(1 for row in result.lineage_rows if row.parent > 0)),
        "frame_count": float(len(result.tracked_masks)),
        "frame_object_count_min": float(min(object_counts) if object_counts else 0),
        "frame_object_count_mean": float(sum(object_counts) / len(object_counts) if object_counts else 0.0),
        "frame_object_count_max": float(max(object_counts) if object_counts else 0),
    }


def finalize_single_tracking_metrics(
    config: TrackingConfig,
    result: TrackingResult,
    output_dir: Path,
    *,
    mask_paths: tuple[Path, ...],
    lineage_path: Path,
) -> None:
    result.output_dir = output_dir
    result.mask_paths = mask_paths
    result.lineage_path = lineage_path
    metrics_payload = {
        "summary_metrics": summarize_tracking_result(result),
        "ctc_evaluation": evaluate_result_with_ctc(
            config.dataset_root / f"{config.track_sequence}_GT",
            output_dir,
        ),
    }
    log_ctc_evaluation("Single-source result", metrics_payload["ctc_evaluation"])
    metrics_json_path = output_dir / "metrics.json"
    metrics_text_path = output_dir / "metrics.txt"
    write_json(metrics_json_path, metrics_payload)
    write_text(metrics_text_path, format_metrics_report(metrics_payload))
    result.metrics = metrics_payload
    result.metrics_json_path = metrics_json_path
    result.metrics_text_path = metrics_text_path
    LOGGER.info("Wrote tracking metrics to %s and %s.", metrics_json_path, metrics_text_path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)
    configure_logging(args.log_level, config.log_file)
    result = run_tracking(config)
    print(f"Tracked sources: {', '.join(result.selected_sources)}")
    if isinstance(result, ConsensusResult):
        print(f"Wrote consensus outputs to {result.output_dir}")
        print(f"Wrote pre-merge metrics to {result.premerge_metrics_path}")
        print(f"Wrote variant comparison to {result.variant_comparison_path}")
        if result.diagnostics_path is not None:
            print(f"Wrote consensus diagnostics to {result.diagnostics_path}")
        for variant_name, evaluation in sorted(result.variant_evaluations.items()):
            print(f"Variant {variant_name}: {len(evaluation.mask_paths)} masks, lineage {evaluation.lineage_path}, metrics {evaluation.metrics_json_path}")
    else:
        print(f"Wrote {len(result.mask_paths)} masks to {result.output_dir}")
        print(f"Wrote lineage file to {result.lineage_path}")
        print(f"Wrote metrics to {result.metrics_json_path}")
    print(f"Wrote run log to {config.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
