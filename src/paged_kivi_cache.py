"""
Combined PagedAttention + KIVI 2-bit KV-Cache.

Merges block-based (paged) memory management with KIVI 2-bit
quantization:

* KV data lives in fixed-size blocks → no fragmentation.
* Completed blocks beyond a "residual window" are quantized to
  2-bit → reduced memory footprint.
* The most recent blocks stay in FP16 for generation quality.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .kivi_cache import (
    QuantizedBlock,
    dequantize,
    quantize_per_channel,
    quantize_per_token,
)


class PagedKIVICache(Cache):
    """Block-based KV-cache with 2-bit quantization for older blocks.

    Parameters
    ----------
    block_size : int
        Tokens per block (page).
    residual_blocks : int
        Number of most-recent *completed* blocks kept in FP16.
    bits : int
        Quantization bit-width for older blocks.
    group_size : int
        Group size for KIVI quantization.
    """

    def __init__(
        self,
        block_size: int = 16,
        residual_blocks: int = 8,
        bits: int = 2,
        group_size: int = 32,
    ):
        super().__init__()
        self.block_size = block_size
        self.residual_blocks = residual_blocks
        self.bits = bits
        self.group_size = group_size

        # FP16 blocks (recent)
        self._key_fp16: List[List[torch.Tensor]] = []
        self._val_fp16: List[List[torch.Tensor]] = []
        self._tokens_in_last: List[int] = []

        # Quantized blocks (older)
        self._key_quant: List[List[QuantizedBlock]] = []
        self._val_quant: List[List[QuantizedBlock]] = []
        self._quant_seq: List[int] = []

        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_fp16):
            return 0
        fp16_len = self._fp16_seq(layer_idx)
        q_len = self._quant_seq[layer_idx] if layer_idx < len(self._quant_seq) else 0
        return q_len + fp16_len

    def get_max_length(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, S, D = key_states.shape

        if layer_idx == 0:
            self._seen_tokens += S

        self._ensure_layer(layer_idx)

        # Write into FP16 blocks (paged style)
        self._write_tokens(key_states, value_states, layer_idx)

        # Quantize old completed blocks beyond the residual window
        self._maybe_quantize(layer_idx)

        return self._get_full_kv(layer_idx)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._key_fp16) <= layer_idx:
            self._key_fp16.append([])
            self._val_fp16.append([])
            self._tokens_in_last.append(0)
            self._key_quant.append([])
            self._val_quant.append([])
            self._quant_seq.append(0)

    def _write_tokens(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        B, H, S, D = key_states.shape
        write_pos = 0

        while write_pos < S:
            blocks_k = self._key_fp16[layer_idx]
            blocks_v = self._val_fp16[layer_idx]

            if not blocks_k or self._tokens_in_last[layer_idx] == self.block_size:
                new_k = torch.zeros(
                    B, H, self.block_size, D,
                    dtype=key_states.dtype, device=key_states.device,
                )
                new_v = torch.zeros(
                    B, H, self.block_size, D,
                    dtype=value_states.dtype, device=value_states.device,
                )
                blocks_k.append(new_k)
                blocks_v.append(new_v)
                self._tokens_in_last[layer_idx] = 0

            pos = self._tokens_in_last[layer_idx]
            space = self.block_size - pos
            to_write = min(space, S - write_pos)

            blocks_k[-1][:, :, pos : pos + to_write, :] = key_states[:, :, write_pos : write_pos + to_write, :]
            blocks_v[-1][:, :, pos : pos + to_write, :] = value_states[:, :, write_pos : write_pos + to_write, :]

            self._tokens_in_last[layer_idx] += to_write
            write_pos += to_write

    def _fp16_seq(self, layer_idx: int) -> int:
        n_blocks = len(self._key_fp16[layer_idx])
        if n_blocks == 0:
            return 0
        return max(0, n_blocks - 1) * self.block_size + self._tokens_in_last[layer_idx]

    def _maybe_quantize(self, layer_idx: int) -> None:
        """Quantize completed FP16 blocks beyond the residual window."""
        blocks_k = self._key_fp16[layer_idx]
        blocks_v = self._val_fp16[layer_idx]

        # Keep at least residual_blocks + 1 (the in-progress one) in FP16
        while len(blocks_k) > self.residual_blocks + 1:
            k_block = blocks_k.pop(0)  # oldest full block
            v_block = blocks_v.pop(0)

            k_data = k_block  # (B, H, block_size, D) — full block
            v_data = v_block

            # Ensure block_size is divisible by group_size
            bs = self.block_size
            usable = (bs // self.group_size) * self.group_size
            if usable == 0:
                # block_size < group_size — store without quantization
                # (fall back to keeping FP16; shouldn't happen with defaults)
                blocks_k.insert(0, k_block)
                blocks_v.insert(0, v_block)
                break

            k_q = quantize_per_channel(k_data[:, :, :usable, :], self.bits, self.group_size)
            v_q = quantize_per_token(v_data[:, :, :usable, :], self.bits, self.group_size)

            self._key_quant[layer_idx].append(k_q)
            self._val_quant[layer_idx].append(v_q)
            self._quant_seq[layer_idx] += usable

            # If there were leftover tokens (usable < block_size), prepend
            if usable < bs:
                leftover_k = k_data[:, :, usable:, :].contiguous()
                leftover_v = v_data[:, :, usable:, :].contiguous()
                blocks_k.insert(0, leftover_k)
                blocks_v.insert(0, leftover_v)

    def _get_full_kv(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []

        # Dequantize older blocks
        for kb in self._key_quant[layer_idx]:
            parts_k.append(dequantize(kb, self.bits))
        for vb in self._val_quant[layer_idx]:
            parts_v.append(dequantize(vb, self.bits))

        # FP16 blocks
        fp16_k = self._key_fp16[layer_idx]
        fp16_v = self._val_fp16[layer_idx]

        for i, (bk, bv) in enumerate(zip(fp16_k, fp16_v)):
            if i == len(fp16_k) - 1:
                # Last block may be partial
                s = self._tokens_in_last[layer_idx]
                parts_k.append(bk[:, :, :s, :])
                parts_v.append(bv[:, :, :s, :])
            else:
                parts_k.append(bk)
                parts_v.append(bv)

        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_usage_bytes(self) -> int:
        total = 0
        for layer_idx in range(len(self._key_fp16)):
            for qb in self._key_quant[layer_idx]:
                total += qb.data.nelement() * qb.data.element_size()
                total += qb.scales.nelement() * qb.scales.element_size()
                total += qb.zeros.nelement() * qb.zeros.element_size()
            for qb in self._val_quant[layer_idx]:
                total += qb.data.nelement() * qb.data.element_size()
                total += qb.scales.nelement() * qb.scales.element_size()
                total += qb.zeros.nelement() * qb.zeros.element_size()
            for blk in self._key_fp16[layer_idx]:
                total += blk.nelement() * blk.element_size()
            for blk in self._val_fp16[layer_idx]:
                total += blk.nelement() * blk.element_size()
        return total

    def __repr__(self) -> str:
        n_layers = len(self._key_fp16)
        seq = self.get_seq_length() if n_layers > 0 else 0
        mem = self.memory_usage_bytes() / (1024 ** 2)
        return (
            f"PagedKIVICache(block_size={self.block_size}, "
            f"residual_blocks={self.residual_blocks}, "
            f"layers={n_layers}, seq={seq}, mem={mem:.1f}MB)"
        )
