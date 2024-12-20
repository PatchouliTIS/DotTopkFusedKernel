#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "kernel_traits.h"
#include "kernel_params.h"
#include "utils.h"
#include "block_info.h"


#define DEBUG


namespace flash {

using namespace cute;


// TODO: Rename ElementAccum
template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN, bool Is_even_K>
__forceinline__ __device__ auto get_topk_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // topK O shape is [batch_size, nums_head, seq_len_q, topk]
        auto gmem_ptr_topk = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_topk_ptr) + bidb * params.o_batch_stride);

        auto topk_shape = make_shape(params.h, params.seqlen_q, params.topk);
        auto topk_stride = make_stride(params.o_head_stride, params.o_row_stride, 1);

        const int tidx = threadIdx.x;

        auto topk_layout = make_layout(topk_shape, topk_stride);
        Tensor mLSE = make_tensor(gmem_ptr_topk, topk_layout);
        auto mLSE_slice = mLSE(bidh, _ , _);
        // TODO: get index in N dimension, May Cause Error
        // return global memory tensor VIEW
        return local_tile(mLSE_slice, Shape<Int<kBlockM>, Int<1>>{}, make_coord(m_block, 0));
}



template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K ,typename Params>
inline __device__ void compute_rowwise_block(const Params &params, const int bidb, const int bidh, const int m_block) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;


    // [batch_size, nums_head, seq_lens, head_dim]
    const BlockInfo</*Varlen=*/false> binfo(params, bidb);
    int n_size = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    // printf("seq_len_q: %d, seq_len_k: %d, d: %d, kBlockM: %d, kBlockN: %d, kHeadDim: %d, kNWarps: %d\n", binfo.actual_seqlen_q, binfo.actual_seqlen_k, params.d,kBlockN, kBlockM, kHeadDim, kNWarps);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // TODO: Early Exit if the block is in front of the diagonal. Have to do in N dimension iteration.
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded;
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, bidb)),
                            make_shape(params.h, binfo.actual_seqlen_q, params.d),
                            make_stride(params.q_head_stride, params.q_row_stride, _1{}));
    
    Tensor gQ = local_tile(mQ(bidh, _, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0)); // [KBlockM, kHeadDim]

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) 
                                            + binfo.k_offset(params.k_batch_stride, bidb)),
                            make_shape(params.h, binfo.actual_seqlen_k, params.d),
                            make_stride(params.k_head_stride, params.k_row_stride, _1{}));
    Tensor gK = local_tile(mK(bidh / params.h_h_k_ratio,_,_), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0)); // [kBlockN, kHeadDim, nblocks]

    // O shape: [batch_size, nums_head, seq_len_q, seq_len_k]
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // Shared Memory
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKV{});


    // gmem tiled copy QK  Gmem --> Shared Memory
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QK = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QK.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QK.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QK.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocks)
    Tensor tKsK = gmem_thr_copy_QK.partition_D(sK);


    // gmem caculation
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);

    Tensor tSgS  = thr_mma.partition_C(gO);
    // clear(tSrQ);
    // clear(tSrK);



    // Copy Atom retiling
    // Smem -> Register for MMA calculation
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);


    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QK.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QK.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));


    // Prologue
    // 
    // clear(tQsQ);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);

    // copy the first sK block 
    // Clear the sK smem tiles since we'll only write out the valid outputs
    // clear(tKsK);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, 0), tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k);
    cute::cp_async_fence();




    // Q*K Dot Product
    #pragma unroll
    for (int step = 0; step < n_size ; ++step) {
        // printf("step: %d\n", step);

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

#ifdef DEBUG
    // printf("acc_s' size: %d, %d, %d\n", size<0>(acc_s), size<1>(acc_s), size<2>(acc_s));
    // printf("acc_s: %d\n", acc_s);
    // printf("acc_s's shape: %d, %d\n", shape<0>(acc_s), shape<1>(acc_s), shape<2>(acc_s));
#endif

        flash::cp_async_wait<0>();
        __syncthreads();

        // BlockQ, BlockK already in shared memory

        flash::gemm</*A_in_regs=*/false>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // printf("acc_s: %d\n", acc_s(0,1,1));
        // write back
        // DO NOT CONVERT TYPE HERE
        // Tensor rP = flash::convert_type<Element>(acc_s);
#ifdef DEBUG
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
            // print(size<2>(tCrA));
            // printf("\n---------------------------------------------\n");
            // printf("--- In compute_rowwise_block AFTER CONVERT TYPE rP RESULT ---\n");
            // printf("\n---------------------------------------------\n");
            // print(acc_s);
            // printf("\n---------------------------------------------\n");
            // print(acc_s.shape());
            // printf("\n---------------------------------------------\n");
            // print(acc_s.size());
            // printf("\n---------------------------------------------\n");
            // print(acc_s.stride());
            // printf("\n---------------------------------------------\n");
            // for(int i = 0 ; i < size<0,0>(acc_s); i++) {
            //     for(int j = 0 ; j < size<0,1>(acc_s); j++) {
            //         for(int k = 0 ; k < size<1>(acc_s); k++) {
            //             for(int l = 0 ; l < size<2>(acc_s); l++) {
            //                 print(acc_s((i,j),k, l));
            //                 printf(" ");
            //             }
            //             printf("\n");
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }
            // printf("\n---------------------------------------------\n");


        //     printf("\n---------------------------------------------\n");
        //     print(rP);
        //     printf("\n---------------------------------------------\n");
        //     print(rP.shape());
        //     printf("\n---------------------------------------------\n");
        //     print(rP.size());
        //     printf("\n---------------------------------------------\n");
        //     print(rP.stride());
        //     printf("\n---------------------------------------------\n");
        //     for(int i = 0 ; i < size<0,0>(rP); i++) {
        //         for(int j = 0 ; j < size<0,1>(rP); j++) {
        //             for(int k = 0 ; k < size<1>(rP); k++) {
        //                 for(int l = 0 ; l < size<2>(rP); l++) {
        //                     print(rP((i,j),k, l));
        //                     printf(" ");
        //                 }
        //                 printf("\n");
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n---------------------------------------------\n");
        // }
#endif
        // Tensor rP_drop = make_fragment_like(rP);
        // directly pass rP to gmem


#ifdef DEBUG
//         if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
//             // print(size<2>(tCrA));
//             printf("\n---------------------------------------------\n");
//             printf("--- In compute_rowwise_block AFTER MAKE FRAGMENT LIKE rP_DROP RESULT ---\n");
//             printf("\n---------------------------------------------\n");
//             print(rP_drop);
//             printf("\n---------------------------------------------\n");
//             print(rP_drop.shape());
//             printf("\n---------------------------------------------\n");
//             print(rP_drop.size());
//             printf("\n---------------------------------------------\n");
//             print(rP_drop.stride());
//             printf("\n---------------------------------------------\n");
//             for(int i = 0 ; i < size<0,0>(rP_drop); i++) {
//                 for(int j = 0 ; j < size<0,1>(rP_drop); j++) {
//                     for(int k = 0 ; k < size<1>(rP_drop); k++) {
//                         for(int l = 0 ; l < size<2>(rP_drop); l++) {
//                             print(rP_drop((i,j),k, l));
//                         }
//                         printf("\n");
//                     }
//                     printf("\n");
//                 }
//                 printf("\n");
//             }
//             printf("\n---------------------------------------------\n");
//         }
#endif
        cute::copy(acc_s, tSgS);
        tSgS.data() = tSgS.data() + kBlockN;



        // 2-level pipeline
        flash::cp_async_wait<0>();
        __syncthreads();

        if(step < n_size - 1) {
            // copy the next sK block
            flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, step + 1), tKsK, tKVcKV, tKVpKV,
                                               binfo.actual_seqlen_k);
            cute::cp_async_fence();
        }
    }

}



template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // printf("bidb: %d, bidh: %d, m_block: %d\n", bidb, bidh, m_block);
    // printf("seqlen_q: %d, seqlen_k: %d, d: %d\n", params.seqlen_q, params.seqlen_k, params.d);

    
    flash::compute_rowwise_block<Kernel_traits, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block);
}

}
