"""Minimal config loader.

The default config uses simple `key: value` YAML syntax so the project does not
need PyYAML. CLI flags remain the source of truth for experiment overrides.
"""

from __future__ import annotations

from pathlib import Path


def _parse_value(value: str) -> object:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    config: dict[str, object] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_value(value)
    return config
