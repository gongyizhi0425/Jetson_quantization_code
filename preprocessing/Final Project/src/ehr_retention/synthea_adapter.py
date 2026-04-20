"""Optional adapter for Synthea CSV exports.

This is intentionally not part of the default pipeline. It lets the project
accept common Synthea CSV files later without changing downstream timeline,
retention, or evaluation code.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_synthea_events(raw_dir: str | Path, limit_patients: int | None = None) -> list[dict[str, object]]:
    base = Path(raw_dir)
    events: list[dict[str, object]] = []
    patient_ids: set[str] = set()

    def include_patient(patient_id: str) -> bool:
        if limit_patients is None or patient_id in patient_ids or len(patient_ids) < limit_patients:
            patient_ids.add(patient_id)
            return True
        return False

    mappings = [
        ("conditions.csv", "diagnosis", "START", "DESCRIPTION", "high"),
        ("medications.csv", "medication_start", "START", "DESCRIPTION", "high"),
        ("procedures.csv", "procedure", "DATE", "DESCRIPTION", "medium"),
        ("encounters.csv", "encounter", "START", "DESCRIPTION", "low"),
    ]
    counter = 1
    for filename, event_type, date_col, text_col, importance in mappings:
        for row in _read_csv(base / filename):
            patient_id = row.get("PATIENT") or row.get("Id") or row.get("PATIENT_ID")
            if not patient_id or not include_patient(patient_id):
                continue
            event_id = f"SYN-{counter:08d}"
            counter += 1
            text = row.get(text_col) or event_type
            events.append(
                {
                    "patient_id": patient_id,
                    "event_id": event_id,
                    "date": (row.get(date_col) or "")[:10],
                    "event_type": event_type,
                    "text": text,
                    "value": text,
                    "importance": importance,
                }
            )
    return events
