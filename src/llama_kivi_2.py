"""
Modern KIVI integration for LLaMA-based models (TinyLlama, OpenLLaMA, LLaMA-2/3).
Supports transformers >= 4.38 (Cache object compatibility) and SDPA for memory efficiency.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

_KIVI_IMPORT_ERROR = None
try:
    from quant.new_pack import triton_quantize_and_pack_along_last_dim
    from quant.matmul import cuda_bmm_fA_qB_outer
except Exception as _e:
    _KIVI_IMPORT_ERROR = _e
    triton_quantize_and_pack_along_last_dim = None
    cuda_bmm_fA_qB_outer = None


class KIVICUDACache(Cache):
    """专属缓存类，用于装载 KIVI C++ 算子生成的底层元组"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kivi_states = []
        self.seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        return None
        
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.seq_len


@dataclass
class LlamaKIVIConfig:
    k_bits: int = 2
    v_bits: int = 2
    group_size: int = 32
    residual_length: int = 128


class LlamaAttention_ModernKIVI(nn.Module):
    def __init__(self, base_attn: nn.Module, cfg: LlamaKIVIConfig):
        super().__init__()
        self.base_attn = base_attn
        self.cfg = cfg

        self.layer_idx = getattr(base_attn, "layer_idx", None)
        if self.layer_idx is None:
            warnings.warn("layer_idx not found! KIVI cache might fail.")

        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj
        self.rotary_emb = getattr(base_attn, "rotary_emb", None)

        self.num_heads = base_attn.config.num_attention_heads
        self.hidden_size = base_attn.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = base_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.k_bits = cfg.k_bits
        self.v_bits = cfg.v_bits
        self.group_size = cfg.group_size
        self.residual_length = cfg.residual_length

    def _pack_kv_initial(self, key_states, value_states):
        if key_states.shape[-2] % self.residual_length != 0:
            if key_states.shape[-2] < self.residual_length:
                key_states_quant, key_states_full = None, key_states
            else:
                split = key_states.shape[-2] % self.residual_length
                key_states_quant = key_states[:, :, :-split, :].contiguous()
                key_states_full = key_states[:, :, -split:, :].contiguous()
        else:
            key_states_quant, key_states_full = key_states, None

        if key_states_quant is not None:
            key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits
            )
        else:
            key_states_quant_trans, key_scale_trans, key_mn_trans = None, None, None

        if value_states.shape[-2] <= self.residual_length:
            value_states_quant, value_states_full, value_scale, value_mn = None, value_states, None, None
        else:
            value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
            value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
            value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                value_states_quant, self.group_size, self.v_bits
            )

        return (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
                value_states_quant, value_states_full, value_scale, value_mn)

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
        output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,**kwargs
    ):
        if cuda_bmm_fA_qB_outer is None:
            raise ImportError("KIVI CUDA ops are unavailable. Please compile kivi_gemv.")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        # 兼容现代 HF Cache
        kivi_past = None
        if use_cache and past_key_value is not None:
            if isinstance(past_key_value, KIVICUDACache):
                while len(past_key_value.kivi_states) <= self.layer_idx:
                    past_key_value.kivi_states.append(None)
                kivi_past = past_key_value.kivi_states[self.layer_idx]

        if kivi_past is not None:
            kv_seq_len += int(kivi_past[-1])

        # 兼容现代 HF 架构：优先使用外层传进来的 position_embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
            
        # 安全应用 RoPE
        try:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        except TypeError:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
        if kivi_past is not None:
            # ==== 解码阶段 (Decode) ====
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, _ = kivi_past

            if key_states_quant_trans is not None:
                key_states_quant_trans_repeat = repeat_kv(key_states_quant_trans, self.num_key_value_groups)
                key_scale_trans_repeat = repeat_kv(key_scale_trans, self.num_key_value_groups)
                key_mn_trans_repeat = repeat_kv(key_mn_trans, self.num_key_value_groups)
                att_qkquant = cuda_bmm_fA_qB_outer(
                    self.group_size, query_states, key_states_quant_trans_repeat,
                    key_scale_trans_repeat, key_mn_trans_repeat, self.k_bits
                )
            else:
                att_qkquant = None

            key_states_full = torch.cat([key_states_full, key_states], dim=2) if key_states_full is not None else key_states
            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))

            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                kq_new, ks_new, km_new = triton_quantize_and_pack_along_last_dim(
                    key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                )
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, kq_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, ks_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, km_new], dim=3)
                else:
                    key_states_quant_trans, key_scale_trans, key_mn_trans = kq_new, ks_new, km_new

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_output = cuda_bmm_fA_qB_outer(
                    self.group_size, attn_weights[:, :, :, :-value_full_length],
                    repeat_kv(value_states_quant, self.num_key_value_groups), 
                    repeat_kv(value_scale, self.num_key_value_groups), 
                    repeat_kv(value_mn, self.num_key_value_groups), self.v_bits
                )
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))

            if value_full_length > self.residual_length:
                vq_new, vs_new, vm_new = triton_quantize_and_pack_along_last_dim(
                    value_states_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits
                )
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, vq_new], dim=2)
                    value_scale = torch.cat([value_scale, vs_new], dim=2)
                    value_mn = torch.cat([value_mn, vm_new], dim=2)
                else:
                    value_states_quant, value_scale, value_mn = vq_new, vs_new, vm_new
        else:
            # ==== 预填充阶段防溢出 (SDPA) ====
            key_states_repeat = repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeat = repeat_kv(value_states, self.num_key_value_groups)
            
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_repeat, value_states_repeat, attn_mask=attention_mask, dropout_p=0.0
                )
            attn_weights = None
            (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
             value_states_quant, value_states_full, value_scale, value_mn) = self._pack_kv_initial(key_states, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if use_cache and isinstance(past_key_value, KIVICUDACache):
            present = (
                key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
                value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len,
            )
            past_key_value.kivi_states[self.layer_idx] = present
            if self.layer_idx == 0:
                past_key_value.seq_len = kv_seq_len

        # 现代 HF Llama 期待返回 3 个值 (包含 cache)
        return attn_output, attn_weights


def patch_llama_with_kivi(model, config_dict: Optional[Dict[str, Any]] = None):
    """注入 KIVI 补丁到 LLaMA 架构模型"""
    if config_dict is None: config_dict = {}
    cfg = LlamaKIVIConfig(
        k_bits=int(config_dict.get("k_bits", 2)),
        v_bits=int(config_dict.get("v_bits", 2)),
        group_size=int(config_dict.get("group_size", 32)),
        residual_length=int(config_dict.get("residual_length", 128))
    )

    n_patched = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            layer.self_attn = LlamaAttention_ModernKIVI(layer.self_attn, cfg)
            n_patched += 1

    print(f"✅ Llama Patched with Modern KIVI! layers={n_patched}, bits={cfg.k_bits}, residual={cfg.residual_length}")
    return model