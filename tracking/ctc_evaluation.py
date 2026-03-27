from __future__ import annotations

import csv
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


LOGGER = logging.getLogger(__name__)
_BC_CANDIDATE_KEYS = ("BC(0)", "BC0", "BC")


def evaluate_result_with_ctc(gt_dir: Path, res_dir: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": "py-ctcmetrics",
        "gt_path": str(gt_dir),
        "res_path": str(res_dir),
    }
    executable = find_ctc_evaluate_executable()
    if executable is None:
        payload["status"] = "skipped"
        payload["reason"] = "ctc_evaluate was not found. Install py-ctcmetrics to enable official CTC evaluation."
        return payload
    if not gt_dir.exists():
        payload["status"] = "skipped"
        payload["reason"] = f"Ground-truth directory does not exist: {gt_dir}"
        return payload
    if not res_dir.exists():
        payload["status"] = "skipped"
        payload["reason"] = f"Result directory does not exist: {res_dir}"
        return payload

    with tempfile.TemporaryDirectory(prefix="ctc_eval_", dir=str(res_dir)) as tmpdir:
        csv_path = Path(tmpdir) / "ctc_metrics.csv"
        command = [
            executable,
            "--gt",
            str(gt_dir),
            "--res",
            str(res_dir),
            "--csv-file",
            str(csv_path),
            "--num-threads",
            "1",
            "--valid",
            "--det",
            "--seg",
            "--tra",
            "--lnk",
            "--ct",
            "--tf",
            "--bc",
            "0",
            "--cca",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        payload["command"] = command
        payload["returncode"] = int(completed.returncode)
        if completed.stdout.strip():
            payload["stdout"] = completed.stdout.strip()
        if completed.stderr.strip():
            payload["stderr"] = completed.stderr.strip()

        if csv_path.exists():
            metrics = _read_metrics_csv(csv_path)
            if metrics:
                payload["status"] = "success"
                payload["metrics"] = _derive_metrics(metrics)
                return payload

        payload["status"] = "failed"
        payload["reason"] = (
            "ctc_evaluate did not produce a readable CSV result."
            if completed.returncode == 0
            else f"ctc_evaluate exited with code {completed.returncode}."
        )
        return payload


def find_ctc_evaluate_executable() -> str | None:
    discovered = shutil.which("ctc_evaluate")
    if discovered is not None:
        return discovered
    sibling = Path(sys.executable).resolve().with_name("ctc_evaluate")
    if sibling.exists():
        return str(sibling)
    return None


def log_ctc_evaluation(label: str, payload: dict[str, object]) -> None:
    status = str(payload.get("status", "unknown"))
    if status == "success":
        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            LOGGER.info(
                "%s official CTC metrics: Valid=%s, DET=%s, SEG=%s, TRA=%s, LNK=%s, BIO=%s, OP_CTB=%s, OP_CLB=%s.",
                label,
                metrics.get("Valid"),
                metrics.get("DET"),
                metrics.get("SEG"),
                metrics.get("TRA"),
                metrics.get("LNK"),
                metrics.get("BIO"),
                metrics.get("OP_CTB"),
                metrics.get("OP_CLB"),
            )
            return
    LOGGER.info(
        "%s official CTC evaluation %s%s",
        label,
        status,
        f": {payload.get('reason')}" if payload.get("reason") else ".",
    )


def _read_metrics_csv(csv_path: Path) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        return {}
    metrics: dict[str, object] = {}
    for key, value in row.items():
        if key is None or value is None or value == "":
            continue
        metrics[key] = _parse_scalar(value)
    return metrics


def _parse_scalar(value: str) -> object:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _derive_metrics(metrics: dict[str, object]) -> dict[str, object]:
    derived = dict(metrics)
    det = _numeric(derived.get("DET"))
    seg = _numeric(derived.get("SEG"))
    tra = _numeric(derived.get("TRA"))
    lnk = _numeric(derived.get("LNK"))
    ct = _numeric(derived.get("CT"))
    tf = _numeric(derived.get("TF"))
    cca = _numeric(derived.get("CCA"))
    bc_key = next((key for key in _BC_CANDIDATE_KEYS if key in derived), None)
    bc = _numeric(derived.get(bc_key)) if bc_key is not None else None
    if det is not None and seg is not None:
        derived["OP_CSB"] = float(0.5 * (det + seg))
    if seg is not None and tra is not None:
        derived["OP_CTB"] = float(0.5 * (seg + tra))
    bio_components = [value for value in (ct, tf, bc, cca) if value is not None]
    if bio_components:
        derived["BIO"] = float(sum(bio_components) / len(bio_components))
    if lnk is not None and _numeric(derived.get("BIO")) is not None:
        derived["OP_CLB"] = float(0.5 * (lnk + float(derived["BIO"])))
    return derived


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
