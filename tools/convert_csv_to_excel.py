from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ConvertResult:
    csv_path: Path
    xlsx_path: Path
    rows: int
    cols: int


def _safe_sheet_name(name: str) -> str:
    # Excel sheet name constraints: <=31 chars and no : \/ ? * [ ]
    bad = ":\\/?*[]"
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip() or "sheet"
    return name[:31]


def convert_one(csv_path: Path, *, overwrite: bool) -> ConvertResult | None:
    if csv_path.suffix.lower() != ".csv":
        return None

    xlsx_path = csv_path.with_suffix(".xlsx")
    if xlsx_path.exists() and not overwrite:
        return None

    df = pd.read_csv(csv_path)
    sheet = _safe_sheet_name(csv_path.stem)

    # Use openpyxl engine by default if installed.
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet)

    return ConvertResult(csv_path=csv_path, xlsx_path=xlsx_path, rows=int(df.shape[0]), cols=int(df.shape[1]))


def iter_csv_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            out.append(root)
            continue
        if root.is_dir():
            out.extend(sorted(root.rglob("*.csv")))
    # de-dupe
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CSV files to .xlsx (one sheet per file).")
    ap.add_argument(
        "paths",
        nargs="*",
        default=["CNN_pruning", "LLM_pruning"],
        help="Files or directories to search (default: CNN_pruning LLM_pruning)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .xlsx")
    ap.add_argument("--delete-csv", action="store_true", help="Delete CSV after successful conversion")
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths]
    csv_files = iter_csv_files(roots)

    converted: list[ConvertResult] = []
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for csv_path in csv_files:
        try:
            res = convert_one(csv_path, overwrite=bool(args.overwrite))
            if res is None:
                skipped += 1
                continue
            converted.append(res)
            if bool(args.delete_csv):
                try:
                    csv_path.unlink()
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001
            failed.append((csv_path, str(e)))

    print(f"Found CSV: {len(csv_files)}")
    print(f"Converted: {len(converted)}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failed)}")

    if converted:
        # Keep this short to avoid noisy output.
        for r in converted[:10]:
            rel_in = os.path.relpath(r.csv_path, Path.cwd())
            rel_out = os.path.relpath(r.xlsx_path, Path.cwd())
            print(f"  {rel_in} -> {rel_out} ({r.rows}x{r.cols})")
        if len(converted) > 10:
            print(f"  ... (+{len(converted) - 10} more)")

    if failed:
        print("\nFailures:")
        for p, msg in failed[:10]:
            print(f"  {p}: {msg}")
        if len(failed) > 10:
            print(f"  ... (+{len(failed) - 10} more)")


if __name__ == "__main__":
    main()
