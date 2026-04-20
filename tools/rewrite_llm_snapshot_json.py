from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _make_repro_snapshot(cfg: Dict[str, Any], calibration_section: str) -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "seed": cfg.get("seed", 0),
        "model": cfg.get("model", {}),
        "eval": cfg.get("eval", {}),
        "pruning": cfg.get("pruning", {}),
        "speed": cfg.get("speed", {}),
        "output": cfg.get("output", {}),
        "run_section": calibration_section,
        "active_calibration": cfg.get(calibration_section, {}),
    }
    if calibration_section != "calibration":
        snap["base_calibration"] = cfg.get("calibration", {})
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite LLM run config snapshots into compact per-run format.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON snapshot (old format)")
    ap.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSON path to write (new format)",
    )
    ap.add_argument(
        "--section",
        choices=["calibration", "calibration_shifted"],
        required=True,
        help="Which calibration section this snapshot corresponds to",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    cfg = json.loads(in_path.read_text(encoding="utf-8"))
    snap = _make_repro_snapshot(cfg, args.section)
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
