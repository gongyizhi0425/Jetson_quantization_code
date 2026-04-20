"""
KIVI 2-bit KV-Cache Implementation.

Based on the KIVI paper: key cache uses per-channel quantization,
value cache uses per-token quantization.  A residual window of
recent tokens is kept in FP16 to preserve generation quality.

Reference: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit
Quantization for KV Cache", 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


# ======================================================================
# Quantized block storage
# ======================================================================

@dataclass
class QuantizedBlock:
    """Packed 2-bit quantized tensor with scales and zero-points."""

    data: torch.Tensor      # uint8, packed (4 values per byte)
    scales: torch.Tensor    # FP16 / BF16
    zeros: torch.Tensor     # FP16 / BF16
    shape: Tuple[int, ...]  # original (B, H, S, D) shape
    quant_dim_size: int     # size of the quantized dimension (before grouping)
    mode: str               # "per_channel" | "per_token"


# ======================================================================
# Bit-packing helpers (2-bit ↔ uint8)
# ======================================================================

def pack_2bit(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor (values in 0-3) → 4 values per byte."""
    flat = tensor.reshape(-1)
    # Pad to multiple of 4
    pad = (4 - flat.shape[0] % 4) % 4
    if pad:
        flat = F.pad(flat, (0, pad))
    flat = flat.reshape(-1, 4)
    packed = (flat[:, 0]
              | (flat[:, 1] << 2)
              | (flat[:, 2] << 4)
              | (flat[:, 3] << 6))
    return packed


def unpack_2bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 → 2-bit values, returning exactly *numel* elements."""
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    flat = torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)
    return flat[:numel]


# ======================================================================
# Quantize / de-quantize
# ======================================================================

def quantize_per_channel(
    tensor: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> QuantizedBlock:
    """Quantize K cache: group along seq-len, stats per channel (head_dim).

    Input shape: ``(B, H, S, D)``
    """
    B, H, S, D = tensor.shape
    max_val = (1 << bits) - 1

    assert S % group_size == 0, (
        f"seq_len {S} must be divisible by group_size {group_size}"
    )
    num_groups = S // group_size

    # (B, H, num_groups, group_size, D)
    x = tensor.reshape(B, H, num_groups, group_size, D)

    x_min = x.amin(dim=3, keepdim=True)   # (B,H,G,1,D)
    x_max = x.amax(dim=3, keepdim=True)

    scale = (x_max - x_min) / max_val
    scale = scale.clamp(min=1e-8)

    x_q = ((x - x_min) / scale).round().clamp(0, max_val).to(torch.uint8)

    packed = pack_2bit(x_q)

    return QuantizedBlock(
        data=packed,
        scales=scale.squeeze(3).to(tensor.dtype),   # (B,H,G,D)
        zeros=x_min.squeeze(3).to(tensor.dtype),
        shape=(B, H, S, D),
        quant_dim_size=group_size,
        mode="per_channel",
    )


def quantize_per_token(
    tensor: torch.Tensor,
    bits: int = 2,
    group_size: int = 32,
) -> QuantizedBlock:
    """Quantize V cache: group along head_dim, stats per token (seq-len).

    Input shape: ``(B, H, S, D)``
    """
    B, H, S, D = tensor.shape
    max_val = (1 << bits) - 1

    # Pad D to multiple of group_size
    pad_d = (group_size - D % group_size) % group_size
    if pad_d:
        tensor = F.pad(tensor, (0, pad_d))
    D_padded = D + pad_d
    num_groups = D_padded // group_size

    # (B, H, S, num_groups, group_size)
    x = tensor.reshape(B, H, S, num_groups, group_size)

    x_min = x.amin(dim=4, keepdim=True)   # (B,H,S,G,1)
    x_max = x.amax(dim=4, keepdim=True)

    scale = (x_max - x_min) / max_val
    scale = scale.clamp(min=1e-8)

    x_q = ((x - x_min) / scale).round().clamp(0, max_val).to(torch.uint8)

    packed = pack_2bit(x_q)

    return QuantizedBlock(
        data=packed,
        scales=scale.squeeze(4).to(tensor.dtype),  # (B,H,S,G)
        zeros=x_min.squeeze(4).to(tensor.dtype),
        shape=(B, H, S, D),       # original D (not padded)
        quant_dim_size=group_size,
        mode="per_token",
    )


def dequantize(block: QuantizedBlock, bits: int = 2) -> torch.Tensor:
    """Dequantize a ``QuantizedBlock`` back to float."""
    B, H, S, D = block.shape

    if block.mode == "per_channel":
        group_size = block.quant_dim_size
        num_groups = S // group_size
        numel = B * H * num_groups * group_size * D
        flat = unpack_2bit(block.data, numel).to(block.scales.device)
        x_q = flat.reshape(B, H, num_groups, group_size, D).float()
        scales = block.scales.unsqueeze(3)   # (B,H,G,1,D)
        zeros = block.zeros.unsqueeze(3)
        x = x_q * scales + zeros
        return x.reshape(B, H, S, D).to(block.scales.dtype)

    elif block.mode == "per_token":
        group_size = block.quant_dim_size
        # Scales shape: (B, H, S, num_groups)
        num_groups = block.scales.shape[-1]
        D_padded = num_groups * group_size
        numel = B * H * S * num_groups * group_size
        flat = unpack_2bit(block.data, numel).to(block.scales.device)
        x_q = flat.reshape(B, H, S, num_groups, group_size).float()
        scales = block.scales.unsqueeze(4)   # (B,H,S,G,1)
        zeros = block.zeros.unsqueeze(4)
        x = x_q * scales + zeros
        x = x.reshape(B, H, S, D_padded)
        return x[:, :, :, :D].to(block.scales.dtype)

    raise ValueError(f"Unknown mode: {block.mode}")


# ======================================================================
# KIVI Cache
# ======================================================================

class KIVICache(Cache):
    """2-bit asymmetric KV-cache following the KIVI scheme.

    * Recent ``residual_length`` tokens are stored in FP16 for quality.
    * Older tokens are quantized to 2-bit (K: per-channel, V: per-token).
    * During attention, quantized blocks are temporarily dequantized.

    Parameters
    ----------
    residual_length : int
        Number of most-recent tokens kept in FP16.
    group_size : int
        Group size for block quantization.
    bits : int
        Quantization bit-width (default 2).
    """

    def __init__(
        self,
        residual_length: int = 128,
        group_size: int = 32,
        bits: int = 2,
    ):
        super().__init__()
        self.residual_length = residual_length
        self.group_size = group_size
        self.bits = bits

        # FP16 residual (recent tokens)
        self._key_residual: List[Optional[torch.Tensor]] = []
        self._value_residual: List[Optional[torch.Tensor]] = []

        # Quantized blocks (older tokens)
        self._key_quant: List[List[QuantizedBlock]] = []
        self._value_quant: List[List[QuantizedBlock]] = []
        self._quant_seq_len: List[int] = []

        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_residual):
            return 0
        res = self._key_residual[layer_idx]
        res_len = res.shape[-2] if res is not None else 0
        q_len = self._quant_seq_len[layer_idx] if layer_idx < len(self._quant_seq_len) else 0
        return q_len + res_len

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
        """Store new K/V and return full (dequantized + residual) for attention."""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Ensure storage for this layer
        while len(self._key_residual) <= layer_idx:
            self._key_residual.append(None)
            self._value_residual.append(None)
            self._key_quant.append([])
            self._value_quant.append([])
            self._quant_seq_len.append(0)

        # Append new tokens to FP16 residual
        if self._key_residual[layer_idx] is None:
            self._key_residual[layer_idx] = key_states
            self._value_residual[layer_idx] = value_states
        else:
            self._key_residual[layer_idx] = torch.cat(
                [self._key_residual[layer_idx], key_states], dim=-2,
            )
            self._value_residual[layer_idx] = torch.cat(
                [self._value_residual[layer_idx], value_states], dim=-2,
            )

        # Quantize overflow
        res_len = self._key_residual[layer_idx].shape[-2]
        if res_len >= self.residual_length + self.group_size:
            self._quantize_overflow(layer_idx)

        # Return full KV for attention
        return self._get_full_kv(layer_idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize_overflow(self, layer_idx: int) -> None:
        k = self._key_residual[layer_idx]
        v = self._value_residual[layer_idx]
        seq = k.shape[-2]

        n_quant = seq - self.residual_length
        n_quant = (n_quant // self.group_size) * self.group_size
        if n_quant <= 0:
            return

        k_part = k[:, :, :n_quant, :].contiguous()
        v_part = v[:, :, :n_quant, :].contiguous()

        k_block = quantize_per_channel(k_part, self.bits, self.group_size)
        v_block = quantize_per_token(v_part, self.bits, self.group_size)

        self._key_quant[layer_idx].append(k_block)
        self._value_quant[layer_idx].append(v_block)
        self._quant_seq_len[layer_idx] += n_quant

        # Trim residual
        self._key_residual[layer_idx] = k[:, :, n_quant:, :].contiguous()
        self._value_residual[layer_idx] = v[:, :, n_quant:, :].contiguous()

    def _get_full_kv(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []

        for kb in self._key_quant[layer_idx]:
            parts_k.append(dequantize(kb, self.bits))
        for vb in self._value_quant[layer_idx]:
            parts_v.append(dequantize(vb, self.bits))

        if self._key_residual[layer_idx] is not None:
            parts_k.append(self._key_residual[layer_idx])
            parts_v.append(self._value_residual[layer_idx])

        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_usage_bytes(self) -> int:
        """Return actual bytes stored in this cache (quantized + residual)."""
        total = 0
        for layer_idx in range(len(self._key_residual)):
            for qb in self._key_quant[layer_idx]:
                total += qb.data.nelement() * qb.data.element_size()
                total += qb.scales.nelement() * qb.scales.element_size()
                total += qb.zeros.nelement() * qb.zeros.element_size()
            for qb in self._value_quant[layer_idx]:
                total += qb.data.nelement() * qb.data.element_size()
                total += qb.scales.nelement() * qb.scales.element_size()
                total += qb.zeros.nelement() * qb.zeros.element_size()
            if self._key_residual[layer_idx] is not None:
                total += self._key_residual[layer_idx].nelement() * self._key_residual[layer_idx].element_size()
                total += self._value_residual[layer_idx].nelement() * self._value_residual[layer_idx].element_size()
        return total

    def __repr__(self) -> str:
        n_layers = len(self._key_residual)
        seq = self.get_seq_length() if n_layers > 0 else 0
        mem = self.memory_usage_bytes() / (1024 ** 2)
        return f"KIVICache(bits={self.bits}, layers={n_layers}, seq={seq}, mem={mem:.1f}MB)"
