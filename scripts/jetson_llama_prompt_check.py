from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_utils import load_prompts
from src.metrics import measure_generation

WORKDIR = Path.cwd().resolve()
DEFAULT_PROMPT_FILE = "ehr_prompts_llama3.2.json"
DEFAULT_HF_CACHE_DIR = ".hf_cache"
DEFAULT_SAVE_DIR = "terminal_checks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a local/cached Llama model on Jetson and run a prompt smoke test."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local model folder, for example unsloth/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help="Prompt JSON file. Relative paths are resolved from the current shell directory.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Which preset prompt to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device. Jetson normally uses cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Weight dtype for model loading.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "kivi"],
        default="baseline",
        help="baseline uses standard Hugging Face KV cache; kivi patches Llama attention with KIVI.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode. Model files must already exist locally or in the Hugging Face cache.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=DEFAULT_HF_CACHE_DIR,
        help="Local Hugging Face cache directory. Relative paths are resolved from the current shell directory.",
    )
    parser.add_argument(
        "--local-model-dir",
        default="",
        help="Optional local model folder. If set and the folder exists, the script loads the model from here first.",
    )
    parser.add_argument(
        "--save-local-copy",
        action="store_true",
        help="After an online download, save a reusable local copy of the model and tokenizer.",
    )
    parser.add_argument(
        "--k-bits",
        type=int,
        default=2,
        help="KIVI key quantization bits.",
    )
    parser.add_argument(
        "--v-bits",
        type=int,
        default=2,
        help="KIVI value quantization bits.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="KIVI group size.",
    )
    parser.add_argument(
        "--residual-length",
        type=int,
        default=128,
        help="KIVI FP16 residual window length.",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory used to save the JSON check result.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-:]", "", text)
    return text


def extract_date(text: str) -> str | None:
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    return match.group(0) if match else None


def simple_checks(reference_answer: str, generated_text: str) -> Dict[str, Any]:
    ref = normalize_text(reference_answer)
    pred = normalize_text(generated_text)
    ref_date = extract_date(reference_answer)
    pred_date = extract_date(generated_text)
    return {
        "reference_normalized": ref,
        "prediction_normalized": pred,
        "exact_match": pred == ref,
        "prediction_contains_reference": ref in pred,
        "same_date": (ref_date == pred_date) if ref_date and pred_date else None,
        "reference_date": ref_date,
        "prediction_date": pred_date,
    }


def sanitize_model_name(model_name_or_path: str) -> str:
    text = model_name_or_path.strip().replace("\\", "/").rstrip("/")
    text = text.split("/")[-1] if "/" in text else text
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or "model"


def resolve_existing_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = WORKDIR / candidate
    return candidate.resolve()


def resolve_prompt_file(path_str: str) -> Path:
    prompt_path = resolve_existing_path(path_str)
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Tip: if you already cd into the target folder, keep {DEFAULT_PROMPT_FILE} beside the script and run it there."
        )
    return prompt_path


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype_name: str,
    device: str,
    offline: bool,
    hf_cache_dir: str,
    local_model_dir: str = "",
    save_local_copy: bool = False,
):
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    cache_dir = resolve_existing_path(hf_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    local_model_path = None
    if local_model_dir:
        local_model_path = resolve_existing_path(local_model_dir)
        if local_model_path.exists():
            model_name_or_path = str(local_model_path)
        elif offline:
            raise FileNotFoundError(
                f"--offline was set but local model folder does not exist: {local_model_path}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=offline,
        cache_dir=str(cache_dir),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=offline,
        cache_dir=str(cache_dir),
    )
    model.eval()

    if save_local_copy and not offline:
        if local_model_path is None:
            local_model_path = WORKDIR / "local_models" / sanitize_model_name(model_name_or_path)
        local_model_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving local model copy to: {local_model_path}")
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path, safe_serialization=True)

    return model, tokenizer, str(cache_dir), model_name_or_path


def build_cache(mode: str, args: argparse.Namespace):
    if mode == "baseline":
        return None

    from src.qwen_kivi_2 import KIVICUDACache, patch_llama_with_kivi

    kivi_config = {
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
    }
    return KIVICUDACache, patch_llama_with_kivi, kivi_config


def main() -> None:
    args = parse_args()

    prompt_file = resolve_prompt_file(args.prompt_file)
    prompts = load_prompts(str(prompt_file))
    if not prompts:
        raise RuntimeError(f"No prompts found in {prompt_file}")
    if args.prompt_index < 0 or args.prompt_index >= len(prompts):
        raise IndexError(f"prompt-index {args.prompt_index} is out of range 0..{len(prompts)-1}")

    item = prompts[args.prompt_index]
    model, tokenizer, hf_cache_dir, resolved_model_source = load_model_and_tokenizer(
        args.model,
        args.dtype,
        args.device,
        args.offline,
        args.hf_cache_dir,
        args.local_model_dir,
        args.save_local_copy,
    )

    cache_impl = None
    if args.mode == "kivi":
        KIVICUDACache, patch_llama_with_kivi, kivi_config = build_cache(args.mode, args)
        model = patch_llama_with_kivi(model, kivi_config)
        cache_impl = KIVICUDACache()

    prompt = item["prompt"]
    num_prompt_tokens = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])

    print("=" * 80)
    print(f"Working dir     : {WORKDIR}")
    print(f"Mode            : {args.mode}")
    print(f"Model           : {args.model}")
    print(f"Model source    : {resolved_model_source}")
    print(f"Offline         : {args.offline}")
    print(f"HF cache dir    : {hf_cache_dir}")
    print(f"Prompt file     : {prompt_file}")
    print(f"Prompt index    : {args.prompt_index}")
    print(f"Prompt tokens   : {num_prompt_tokens}")
    print(f"Question        : {item.get('question', '')}")
    print(f"Reference answer: {item.get('reference_answer', '')}")
    print("=" * 80)

    warmup_cache = None if args.mode == "baseline" else type(cache_impl)()
    _ = measure_generation(
        model,
        tokenizer,
        prompt,
        max_new_tokens=min(16, args.max_new_tokens),
        cache_impl=warmup_cache,
        device=args.device,
    )
    torch.cuda.empty_cache()

    cache_for_run = None if args.mode == "baseline" else type(cache_impl)()
    metrics = measure_generation(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        cache_impl=cache_for_run,
        device=args.device,
    )

    checks = simple_checks(item.get("reference_answer", ""), metrics.generated_text)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": args.model,
        "model_source": resolved_model_source,
        "offline": args.offline,
        "hf_cache_dir": hf_cache_dir,
        "prompt_file": str(prompt_file),
        "prompt_index": args.prompt_index,
        "question": item.get("question", ""),
        "reference_answer": item.get("reference_answer", ""),
        "generated_text": metrics.generated_text,
        "metrics": {
            "ttft_ms": metrics.ttft_ms,
            "tpot_ms": metrics.tpot_ms,
            "total_time_ms": metrics.total_time_ms,
            "num_input_tokens": metrics.num_input_tokens,
            "num_output_tokens": metrics.num_output_tokens,
            "peak_memory_mb": metrics.peak_memory_mb,
            "model_weight_mb": metrics.model_weight_mb,
            "kv_cache_memory_mb": metrics.kv_cache_memory_mb,
            "memory_fragmentation": metrics.memory_fragmentation,
            "memory_utilization": metrics.memory_utilization,
        },
        "checks": checks,
    }

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.mode}_prompt_{args.prompt_index:03d}.json"
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n--- Generation metrics ---")
    print(f"TTFT            : {metrics.ttft_ms:.1f} ms")
    print(f"TPOT            : {metrics.tpot_ms:.1f} ms")
    print(f"Peak memory     : {metrics.peak_memory_mb:.1f} MB")
    print(f"KV cache memory : {metrics.kv_cache_memory_mb:.1f} MB")
    print(f"Output tokens   : {metrics.num_output_tokens}")
    print("\n--- Simple checks ---")
    print(f"Exact match               : {checks['exact_match']}")
    print(f"Prediction contains ref   : {checks['prediction_contains_reference']}")
    print(f"Same date                 : {checks['same_date']}")
    print("\n--- Generated text ---")
    print(metrics.generated_text)
    print(f"\nSaved JSON report to: {save_path}")


if __name__ == "__main__":
    main()
