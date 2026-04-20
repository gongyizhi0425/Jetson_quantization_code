"""Build patient timelines from event records."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def sort_events(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(events, key=lambda e: (str(e["patient_id"]), str(e["date"]), str(e["event_id"])))


def event_to_line(event: dict[str, object]) -> str:
    return f"{event['date']} | {event['event_type']} | {event['text']} [event_id={event['event_id']}]"


def build_timelines(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in sort_events(events):
        grouped[str(event["patient_id"])].append(event)

    timelines = []
    for patient_id in sorted(grouped):
        patient_events = grouped[patient_id]
        timelines.append(
            {
                "patient_id": patient_id,
                "events": patient_events,
                "timeline_text": "\n".join(event_to_line(e) for e in patient_events),
            }
        )
    return timelines


def validate_timeline(timeline: dict[str, object]) -> None:
    events = timeline.get("events", [])
    if not isinstance(events, list) or not events:
        raise ValueError("timeline has no events")
    dates = [str(event["date"]) for event in events]
    if dates != sorted(dates):
        raise ValueError(f"events are not sorted for {timeline.get('patient_id')}")
    event_ids = [event.get("event_id") for event in events]
    if any(not event_id for event_id in event_ids):
        raise ValueError(f"missing event_id for {timeline.get('patient_id')}")
