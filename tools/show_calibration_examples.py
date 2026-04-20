from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isabs(path) and not os.path.exists(path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, path)
        if os.path.exists(candidate):
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _short(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _print_token_preview(tok: AutoTokenizer, ids: list[int], limit: int = 24) -> None:
    preview = ids[:limit]
    text = tok.decode(preview, skip_special_tokens=False)
    print(f"  token_ids[:{limit}]= {preview}")
    print(f"  decoded[:{limit}]= {_short(text, 220)}")


def _show_wikitext_examples(tok: AutoTokenizer, *, split: str, config: str, n: int, seq_len: int) -> None:
    ds = load_dataset("wikitext", config, split=split)
    key = "text" if "text" in ds.column_names else ds.column_names[0]

    picked = 0
    i = 0
    while picked < n and i < len(ds):
        text = str(ds[i][key])
        i += 1
        if not text.strip():
            continue
        enc = tok(text, add_special_tokens=False)
        ids = enc["input_ids"]
        if len(ids) < 8:
            continue
        # Slice to seq_len for a comparable preview
        ids = ids[:seq_len]
        print(f"- wikitext sample #{picked+1} (len_tokens={len(ids)})")
        print(f"  text: {_short(text)}")
        _print_token_preview(tok, ids)
        picked += 1


def _show_random_token_examples(tok: AutoTokenizer, *, n: int, seq_len: int, seed: int) -> None:
    rng = random.Random(seed)
    vocab = int(tok.vocab_size)
    for k in range(n):
        ids = [rng.randrange(0, vocab) for _ in range(seq_len)]
        print(f"- random_tokens sample #{k+1} (len_tokens={len(ids)})")
        _print_token_preview(tok, ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Show concrete examples of wikitext calibration vs random_tokens.")
    ap.add_argument("--config", required=True, help="Path to llm_wanda.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="Number of samples to print")
    ap.add_argument("--seq-len", type=int, default=128, help="Token length to preview")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name"))
    cache_dir = args.cache_dir or model_cfg.get("cache_dir")
    local_only = bool(args.local_files_only or model_cfg.get("local_files_only", False))

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir, local_files_only=local_only)

    calib = cfg.get("calibration", {})
    shifted = cfg.get("calibration_shifted", {})

    print("=== Tokenizer ===")
    print(json.dumps({"model": model_name, "vocab_size": int(tok.vocab_size)}, indent=2, ensure_ascii=False))

    print("\n=== calibration (wikitext) examples ===")
    _show_wikitext_examples(
        tok,
        split=str(calib.get("split", "train")),
        config=str(calib.get("config", "wikitext-2-raw-v1")),
        n=int(args.n),
        seq_len=int(args.seq_len),
    )

    if bool(shifted.get("enabled", False)):
        print("\n=== calibration_shifted examples ===")
        ds_name = str(shifted.get("dataset", ""))
        if ds_name == "random_tokens":
            _show_random_token_examples(tok, n=int(args.n), seq_len=int(args.seq_len), seed=int(args.seed))
        else:
            print(f"(shifted dataset is not random_tokens: {ds_name})")


if __name__ == "__main__":
    main()
