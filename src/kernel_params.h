/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <vector>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;

    // The stride between rows of the Q, K and topk_O matrices.
    // shape -> [batch_size, nheads, seq_lens, head_dim]
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    void *__restrict__ ido_ptr;     // [batch_size, num_heads, seq_len_q, topk]

    index_t ido_batch_stride;
    index_t ido_head_stride;
    index_t ido_row_stride;
    
    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q, topk;

    // array of length b+1 holding starting offset of each sequence.
    // int * __restrict__ cu_seqlens_q;
    // int * __restrict__ cu_seqlens_k;
    // int * __restrict__ leftpad_k;  /* for varlen */

    // int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    // void * __restrict__ knew_ptr;
    // void * __restrict__ vnew_ptr; /* for kvcache decode phase */

    // The stride between rows of the Q, K and V matrices.
    // index_t knew_batch_stride;
    // index_t vnew_batch_stride;
    // index_t knew_row_stride;
    // index_t vnew_row_stride;
    // index_t knew_head_stride;
    // index_t vnew_head_stride;

    bool is_bf16;
    bool is_causal;
};