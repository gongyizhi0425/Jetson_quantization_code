"""
Simplified PagedAttention KV-Cache.

Uses fixed-size blocks (pages) for KV storage instead of one large
contiguous tensor.  Blocks are allocated on-demand, eliminating memory
fragmentation.

Note
----
A production PagedAttention (vLLM) uses custom CUDA kernels to compute
attention directly on paged memory.  This implementation gathers pages
into a contiguous tensor for the standard HF attention path, but still
demonstrates the *memory-management* benefits.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class PagedKVCache(Cache):
    """KV-Cache with block-based (paged) memory allocation.

    Parameters
    ----------
    block_size : int
        Number of tokens stored per block (page).
    """

    def __init__(self, block_size: int = 16):
        super().__init__()
        self.block_size = block_size

        # Per-layer lists of blocks.  Each block is a tensor of shape
        # (B, num_kv_heads, block_size, head_dim).
        self._key_blocks: List[List[torch.Tensor]] = []
        self._val_blocks: List[List[torch.Tensor]] = []
        self._tokens_in_last: List[int] = []

        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._key_blocks) or not self._key_blocks[layer_idx]:
            return 0
        n_full = max(0, len(self._key_blocks[layer_idx]) - 1) * self.block_size
        return n_full + self._tokens_in_last[layer_idx]

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

        # Ensure layer storage exists
        while len(self._key_blocks) <= layer_idx:
            self._key_blocks.append([])
            self._val_blocks.append([])
            self._tokens_in_last.append(0)

        # Write tokens into blocks
        write_pos = 0
        while write_pos < S:
            blocks_k = self._key_blocks[layer_idx]
            blocks_v = self._val_blocks[layer_idx]

            if not blocks_k or self._tokens_in_last[layer_idx] == self.block_size:
                # Allocate a new block
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

            # Fill current block
            pos = self._tokens_in_last[layer_idx]
            space = self.block_size - pos
            to_write = min(space, S - write_pos)

            blocks_k[-1][:, :, pos : pos + to_write, :] = key_states[:, :, write_pos : write_pos + to_write, :]
            blocks_v[-1][:, :, pos : pos + to_write, :] = value_states[:, :, write_pos : write_pos + to_write, :]

            self._tokens_in_last[layer_idx] += to_write
            write_pos += to_write

        return self._gather(layer_idx)

    # ------------------------------------------------------------------
    # Gather blocks → contiguous
    # ------------------------------------------------------------------

    def _gather(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks_k = self._key_blocks[layer_idx]
        blocks_v = self._val_blocks[layer_idx]

        if not blocks_k:
            raise RuntimeError("No blocks allocated yet")

        if len(blocks_k) == 1:
            s = self._tokens_in_last[layer_idx]
            return blocks_k[0][:, :, :s, :], blocks_v[0][:, :, :s, :]

        parts_k = list(blocks_k[:-1])
        parts_v = list(blocks_v[:-1])

        s = self._tokens_in_last[layer_idx]
        parts_k.append(blocks_k[-1][:, :, :s, :])
        parts_v.append(blocks_v[-1][:, :, :s, :])

        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def memory_usage_bytes(self) -> int:
        total = 0
        for layer_blocks in self._key_blocks:
            for blk in layer_blocks:
                total += blk.nelement() * blk.element_size()
        for layer_blocks in self._val_blocks:
            for blk in layer_blocks:
                total += blk.nelement() * blk.element_size()
        return total

    def num_allocated_blocks(self) -> int:
        return sum(len(layer) for layer in self._key_blocks)

    def __repr__(self) -> str:
        n_layers = len(self._key_blocks)
        seq = self.get_seq_length() if n_layers > 0 else 0
        blks = self.num_allocated_blocks()
        mem = self.memory_usage_bytes() / (1024 ** 2)
        return (
            f"PagedKVCache(block_size={self.block_size}, layers={n_layers}, "
            f"seq={seq}, blocks={blks}, mem={mem:.1f}MB)"
        )
