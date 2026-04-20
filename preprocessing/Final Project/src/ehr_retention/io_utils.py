"""Small JSONL and filesystem helpers used by scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping


JsonRecord = MutableMapping[str, object]


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> Iterator[JsonRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> int:
    out = ensure_parent(path)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count
