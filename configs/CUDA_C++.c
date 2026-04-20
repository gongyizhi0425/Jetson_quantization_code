// KIVI fused kernel 的核心思路 (简化)
__global__ void fused_dequant_attention(
    uint8_t* packed_K,    // 2-bit packed, 很小
    uint8_t* packed_V,    
    half* Q,              // query
    half* scales_K, half* zeros_K,
    half* output,
    int seq_len, int head_dim
) {
    // 1. 在寄存器中解包 2-bit → FP16
    half k_val = dequant_2bit(packed_K[idx], scales_K[g], zeros_K[g]);
    
    // 2. 直接在寄存器中算 Q·K (不写回显存)
    float score = 0;
    for (int d = 0; d < head_dim; d++)
        score += (float)Q[d] * (float)k_val[d];
    
    // 3. shared memory 做 softmax
    __shared__ float scores[MAX_SEQ];
    scores[tid] = score;
    __syncthreads();
    softmax_inplace(scores);
    
    // 4. 直接用解包后的 V 加权, 只写最终结果
    half v_val = dequant_2bit(packed_V[idx], scales_V[g], zeros_V[g]);
    output[d] += scores[tid] * v_val;
}