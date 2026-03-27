from __future__ import annotations

import json
from pathlib import Path


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
