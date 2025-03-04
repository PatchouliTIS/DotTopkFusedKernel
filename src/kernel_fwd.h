#pragma once

#include "cute/tensor.hpp"
#include <cmath>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include "kernel_traits.h"
#include "kernel_params.h"
#include "utils.h"
#include "block_info.h"
#include "topk.h"


#define DEBUG


namespace flash {

using namespace cute;



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
    constexpr int kTopk = Kernel_traits::topK;


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


    // TODO: remove gO from kernel, no need to store it in gmem
    // O shape: [batch_size, nums_head, seq_len_q, seq_len_k]
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    // IDO shape: [batch_size, nums_head, seq_len_q, topk]  for topk index output
    // Every Block of IDO is [kBlockM, kTopk]
    const index_t row_offset_ido = ((bidb * params.h + bidh) * params.seqlen_q_rounded + m_block * kBlockM) * params.topk;
//     Tensor gIDO = make_tensor(make_gmem_ptr(reinterpret_cast<index_t*>(params.ido_ptr) + row_offset_ido),
//                             Shape<Int<kBlockM>, Int<kTopk>>{},
//                             make_stride(kTopk, _1{}));
    
// #ifdef DEBUG
//     if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
//         printf("\n---------------------------------------------\n");
//         printf("--- In compute_rowwise_block RESULT gIDO ---\n");
//         printf("\n---------------------------------------------\n");
//         print_tensor(gIDO);
//         printf("\n---------------------------------------------\n");
//     }
// #endif



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

    
    // generate thread-level coordinate tensor acc_i on REGs for TOPK
    Tensor tIrI = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});                     // MMA=(2,2) , MMA_M=1, MMA_N=8
    Tensor global_value = make_tensor(tIrI.data(), flash::convert_layout_acc_rowcol(tIrI.layout()));        // ((2, MMA_M), (2, MMA_N))
    // clear(acc_i);
    Tensor global_index = make_tensor_like<index_t>(global_value);
    // int strideInThr = size<2>(tSgS);  // 8 According to TiledMMA.Layout_C
    // int strideAmongThr = 32 >> 4;   // layout<0,0>TiledMMA.Layout_C / Atom_MMA_M

    flash::TopK<size<0>(global_value), size<1>(global_value), 3, 1, kBlockN, Element, index_t> topk;
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("\n---------------------------------------------\n");
        printf("--- TOPK PARAMS CHECK ---\n");
        printf("\n---------------------------------------------\n");
        printf("strideBitInThr: %d, strideBitAmongThr: %d\n", 3, 1);
        printf("\n---------------------------------------------\n");
        printf("global_index:\n");
        print_tensor(global_index);
        printf("\n---------------------------------------------\n");    
        printf("tIrI:\n");
        print_tensor(tIrI);
        printf("\n---------------------------------------------\n");
        printf("global_value:\n");
        print_tensor(global_value);
        printf("\n---------------------------------------------\n");
    }
#endif

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

#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("\n---------------------------------------------\n");
        printf("--- In compute_rowwise_block PROLOGUE ---\n");
        printf("\n---------------------------------------------\n");
        printf("gmem_tiled_copy_QKV:\n");
        print(gmem_tiled_copy_QKV);
        printf("\nKernel_traits::kBlockKSmem: ");
        print(Kernel_traits::kBlockKSmem);
        printf("\nKernel_traits::kGmemElemsPerLoad: ");
        print(Kernel_traits::kGmemElemsPerLoad);
        printf("\nKernel_traits::kNThreads: ");
        print(Kernel_traits::kNThreads);
        printf("\nKernel_traits::SmemCopyAtom: ");
        print(typename Kernel_traits::SmemCopyAtom{});
        printf("\nsmem_tiled_copy_Q: ");
        print(smem_tiled_copy_Q);
        // printf("\ngQ:\n");
        // print_tensor(gQ);
        printf("\ntQgQ:\n");
        print_tensor(tQgQ);
        printf("\ntQcQ:\n");
        print_tensor(tQcQ);
        printf("\ntQpQ:\n");
        print_tensor(tQpQ);
        printf("\ntiled_mma:\n");
        print(tiled_mma);
        printf("\ntiled_mma->layoutA_TV:\n");
        print(tiled_mma.get_layoutA_TV());
        printf("\ntiled_mma->layoutA_MK:\n");
        print(tiled_mma.get_layoutA_MK());
        printf("\ntiled_copy A shape:\n");
        print(make_shape(tile_size<0>(tiled_mma),tile_size<2>(tiled_mma)));
        printf("\ntSrQ:\n");
        print(tSrQ);
        printf("\ntSsQ: \n");
        print(tSsQ);
        printf("\ngO:\n");
        // print_tensor(gO);
        // printf("\ntSgS: \n");
        // print_tensor(tSgS);
        printf("\n---------------------------------------------\n");
    }
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1) {
        printf("\n---------------------------------------------\n");
        printf("--- In compute_rowwise_block PROLOGUE THREAD 1---\n");
        printf("\n---------------------------------------------\n");
        // printf("tSgS: \n");
        // print_tensor(tSgS);
        printf("\n---------------------------------------------\n");
    }
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 20) {
        printf("\n---------------------------------------------\n");
        printf("--- In compute_rowwise_block PROLOGUE THREAD 2 ---\n");
        printf("\n---------------------------------------------\n");
        // printf("tSgS: \n");
        // print_tensor(tSgS);
        printf("\n---------------------------------------------\n");
    }
#endif


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
    // launch async copy pipeline
    cute::cp_async_fence();




    // Q*K Dot Product
    #pragma unroll
    for (int step = 0; step < n_size ; ++step) {
        // printf("step: %d\n", step);

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);

        // 2-level pipeline according to cute::cp_async_fence()
        flash::cp_async_wait<0>();
        __syncthreads();

        // BlockQ, BlockK already in shared memory
        flash::gemm</*A_in_regs=*/false>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        


#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
            printf("\n---------------------------------------------\n");
            printf("--- In compute_rowwise_block RESULT acc_s OUTPUTS THREAD 0---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(acc_s);
            printf("\n---------------------------------------------\n");
        }


        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1) {
            printf("\n---------------------------------------------\n");
            printf("--- In compute_rowwise_block RESULT acc_s OUTPUTS THREAD 1---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(acc_s);
            printf("\n---------------------------------------------\n");
        }
#endif
//         // directly pass value into gmem
        cute::copy(acc_s, tSgS);
        tSgS.data() = tSgS.data() + kBlockN;


#ifdef DEBUG
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
        //     printf("\n---------------------------------------------\n");
        //     printf("--- In compute_rowwise_block RESULT tSgS THREAD 2 ---\n");
        //     printf("\n---------------------------------------------\n");
        //     print(tSgS);
        //     printf("\n");
        //     print_tensor(tSgS);
        //     printf("\n");
        //     print(tSgS.layout());
        //     printf("\n---------------------------------------------\n");
        // }

        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 1) {
        //     printf("\n---------------------------------------------\n");
        //     printf("--- In compute_rowwise_block RESULT tSgS THREAD 1 ---\n");
        //     printf("\n---------------------------------------------\n");
        //     print(tSgS);
        //     printf("\n");
        //     print_tensor(tSgS);
        //     printf("\n");
        //     print(tSgS.layout());
        //     printf("\n---------------------------------------------\n");
        // }
#endif



        // 2-level pipeline
        flash::cp_async_wait<0>();
        __syncthreads();

        if(step < n_size - 1) {
            // copy the next sK block
            flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, step + 1), tKsK, tKVcKV, tKVpKV,
                                               binfo.actual_seqlen_k);
            cute::cp_async_fence();
        }


        // TOPK
        (step == 0)
            ? topk.template topk_index<true, index_t>(acc_s, global_index, global_value, tidx, step )
            : topk.template topk_index<false, index_t>(acc_s, global_index, global_value, tidx, step);
    }

#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 20) {
        printf("\n---------------------------------------------\n");
        printf("--- In TopK ROW COMPLETE RESULT global_index THREAD 20 ---\n");
        printf("\n---------------------------------------------\n");
        print_tensor(global_index);
        printf("\n---------------------------------------------\n");
    }
#endif


    // Copy IDO to gmem

    // First, do warp-shuffle to dispatch values to each thread.
    // WHY WE DO THAT?
    // After collect the topk index from each thread, the first thread of very 4 threads (cuz we do a 16-size bitonic topk among 64 values in a row) within a warp has the total data, which should be copied to gmem.
    // The mapping are demostrated as below:
    // thread 0 global_index in shape (2, 16) (converted from ((2,2),1,8))
    // thread coordinate --->  gIDO gmem coordinate (gmem shape is (block_m=64,topk=16))
    //      ( 0 , 0 )   ---->  ( 0 , 0 )      ( 0 , 1 )   ---->  ( 0 , 1 )   ( 0 , 2 )   ---->  ( 0 , 2 ) .....   ( 0 , 14 ) ----> ( 0, 14 )   ( 0, 15 ) ----> ( 0, 15 )
    //      ( 1 , 0 )   ---->  ( 8 , 0 )      ( 1 , 1 )   ---->  ( 8 , 1 )   ( 1 , 2 )   ---->  ( 8 , 2 ) .....   ( 1 , 14 ) ----> ( 8, 14 )   ( 1, 15 ) ----> ( 8, 15 )
    //                           ^ this STRIDE or LAYOUT also defined at MMA_Atom.Layout_C: LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
    //                                                                                                                                    ^                                                                

    // 1. init final tensor
//     Tensor tIgI = make_tensor<index_t>(make_shape(_1{}, _8{}));
//     clear(tIgI);
//     if((tidx & 0x11) == 0) {
//         #pragma unroll
//         for(int i = 0; i < 8; i++) {
//             tIgI(0,i) = global_index(0, i);
//         }
//     }

//     // 2. warp-shuffle to dispatch values to each thread
//     // T0 --> T1
//     #pragma unroll
//     for(int i = 8; i < 16; i++) {
//         warp_scatter_index(global_index(0, i), 1, 4, 0x88888888);
//     }
//     if((tidx & 0x11) == 1) {
//         #pragma unroll
//         for(int i = 8; i < 16; i++) {
//             tIgI(0,i-8) = global_index(0, i);
//         }
//     }

//     // T0 --> T2
//     #pragma unroll
//     for(int i = 0; i < 8; i++) {
//         warp_scatter_index(global_index(1, i),  2, 4, 0x88888888);
//     }
//     if((tidx & 0x11) == 2) {
//         #pragma unroll
//         for(int i = 0; i < 8; i++) {
//             tIgI(0,i) = global_index(1, i);
//         }
//     }

//     // T0 --> T3
//     #pragma unroll
//     for(int i = 8; i < 16; i++) {
//         warp_scatter_index(global_index(1, i), 3, 4, 0x88888888);
//     }
//     if((tidx & 0x11) == 3) {
//         #pragma unroll
//         for(int i = 8; i < 16; i++) {
//             tIgI(0,i-8) = global_index(1, i);
//         }
//     }

// #ifdef DEBUG
//     if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
//         printf("\n---------------------------------------------\n");
//         printf("--- In TopK ROW COMPLETE RESULT tIgI ---\n");
//         printf("\n---------------------------------------------\n");
//         print_tensor(tIgI);
//         printf("\n---------------------------------------------\n");
//         printf("--- In TopK ROW COMPLETE RESULT global_index ---\n");
//         printf("\n---------------------------------------------\n");
//         print_tensor(global_index);
//         printf("\n---------------------------------------------\n");
//     }
// #endif


    // 2. init Tiled_Copy
    // typename Kernel_traits::GmemTiledCopyIDO gmem_tiled_copy_IDO;
    // auto gmem_thr_copy_IDO = gmem_tiled_copy_IDO.get_thread_slice(tidx);
    // Tensor tIgI_D = gmem_thr_copy_IDO.partition_D(gIDO);

    // cute::copy(gmem_tiled_copy_IDO,tIgI, tIgI_D);

    const index_t warp_id = (tidx >> (0x5));
    const index_t lane_id = (tidx & (0x1f));

#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("\n---------------------------------------------\n");
        printf("--- In TopK ROW COMPLETE RESULT global_index ---\n");
        printf("\n---------------------------------------------\n");
        print_tensor(global_index);
        printf("\n---------------------------------------------\n");
    }
#endif


    #pragma unroll
    for(int i = 0; i < 16; i++) {
        warp_scatter_index(global_index(1, i), 2, 4, 0x55555555);
    }


#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
        printf("\n---------------------------------------------\n");
        printf("--- In TopK ROW COMPLETE RESULT global_index THREAD 2 ---\n");
        printf("\n---------------------------------------------\n");
        for(int i = 0; i < size<0>(global_index); i++) {
            for(int j = 0; j < size<1>(global_index); j++) {
                printf("global_index(%d, %d): %d\n", i, j, global_index(i, j));
            }
        }
        printf("\n---------------------------------------------\n");
    }
#endif

    if ((tidx & 0x1) == 0) {
        if((tidx & 0x2) == 0) {
            // Tensor tIgI_S = make_tensor<index_t>(make_shape(_1{}, _16{}));
            Tensor tIrI_S = local_tile(global_index, Shape<_1, Int<kTopk>>{}, make_coord((0, 0)));  //  make_coord((0, 0))
            // Tensor tIgI_S = tiled_divide(global_index, Shape<_1, Int<kTopk>>{});
            // Tensor gIDO = make_tensor(make_gmem_ptr(reinterpret_cast<index_t*>(params.ido_ptr) + row_offset_ido),
            //                 Shape<Int<kBlockM>, Int<kTopk>>{},
            //                 make_stride(kTopk, _1{}));
            Tensor tIgI_D = make_tensor(make_gmem_ptr(reinterpret_cast<index_t*>(params.ido_ptr) + (row_offset_ido + ((warp_id << (0x4)) + (lane_id >> 2)) * params.topk)),
                            Shape<_1, Int<kTopk>>{},
                            make_stride(kTopk, _1{}));
            // Tensor tIgI_D = local_tile(global_index, Shape<_1, Int<kTopk>>{},make_coord(, 0)));  //  make_coord(((warp_id << (0x4)) + (lane_id >> 2), 0))
#ifdef DEBUG
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
                printf("\n------------------------IN SECTION 1------------------------------\n");
                printf("\n--------------------THREAD 0 BEFORE COPY-------------------------\n");
                printf("tIrI_S:\n");
                print_tensor(tIrI_S);
                printf("\n---------------------------------------------\n");
            }
#endif
            cute::copy(tIrI_S, tIgI_D);
#ifdef DEBUG
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
                printf("\n------------------------IN SECTION 1------------------------------\n");
                printf("\n--------------------THREAD 0 AFTER COPY-------------------------\n");
                printf("tIrI_S:\n");
                print_tensor(tIrI_S);
                printf("tIgI_D:\n");
                print_tensor(tIgI_D);
                printf("\n---------------------------------------------\n");
            }
#endif
        } else {
            // Tensor tIgI_S = local_tile(global_index, Shape<_1, Int<kTopk>>{}, make_coord((1, 0)));
            Tensor tmpReg = tiled_divide(global_index, Shape<_1, Int<kTopk>>{});
            Tensor tIrI_S = tmpReg(make_coord(_,_), 1, 0);
            Tensor tIgI_D = make_tensor(make_gmem_ptr(reinterpret_cast<index_t*>(params.ido_ptr) + (row_offset_ido + ((warp_id << 0x4) + ((lane_id - 2) >> 2) + 8) * params.topk)),   // 
                            Shape<_1, Int<kTopk>>{},
                            make_stride(kTopk, _1{}));
#ifdef DEBUG
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
                printf("\n------------------------IN SECTION 2------------------------------\n");
                printf("\n--------------------THREAD 2 BEFORE COPY-------------------------\n");
                printf("tIrI_S:\n");
                print_tensor(tIrI_S);
                printf("tmpReg:\n");
                print_tensor(tmpReg);
                printf("\n---------------------------------------------\n");
            }
#endif
            // Tensor tIgI_D = local_tile(global_index, Shape<_1, Int<kTopk>>{}, make_coord(((warp_id << (0x4)) + (((lane_id - 2) >> 2)) + 8, 0)));
            cute::copy(tIrI_S, tIgI_D);
#ifdef DEBUG
            if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
                printf("\n------------------------IN SECTION 2------------------------------\n");
                printf("\n--------------------THREAD 2 AFTER COPY-------------------------\n");
                printf("tIrI_S:\n");
                print_tensor(tIrI_S);
                printf("tIgI_D:\n");
                print_tensor(tIgI_D);
                printf("global_index:\n");
                print_tensor(global_index);
                printf("\n---------------------------------------------\n");
            }
#endif
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
