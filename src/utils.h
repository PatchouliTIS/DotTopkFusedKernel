#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>
#include <cmath>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// #define DEBUG

namespace flash {
template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
        typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
        typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Thr_tile Bitonic Sort
template<typename IDX>
__forceinline__ __device__  void warp_scatter_index(IDX &idx, const int &offset, const int &stride, unsigned int mask = 0xffffffff) {
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
        printf("\n---------------------------------------------\n");
        printf("--- In warp_scatter_index THREAD 2---\n");
        printf("\n---------------------------------------------\n");
        printf("idx: %d, offset: %d, stride: %d, mask: %x\n", idx, offset, stride, mask);
        printf("\n---------------------------------------------\n");
    }
#endif
    idx = __shfl_up_sync(mask, idx, offset, stride);
    return;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("\n---------------------------------------------\n");
        printf("--- In gemm PREPROCESS ---\n");
        printf("tCsA Total:\n");
        print_tensor(tCsA);
        print(tCsA.layout());
        printf("\n---------------------------------------------\n");
    }
#endif

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
#ifdef DEBUG
if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        // print(size<2>(tCrA));
        printf("\n---------------------------------------------\n");
        printf("--- In gemm ITERATION ---\n");
        printf("Current inner gemm iteration times: %d\n", i);
        // printf("\n---------------------------------------------\n");
        // print(acc);
        // printf("\n---------------------------------------------\n");
        // print(acc.shape());
        // printf("\n---------------------------------------------\n");
        // print(acc.size());
        // printf("\n---------------------------------------------\n");
        // print(acc.stride());
        // printf("\n---------------------------------------------\n");
        // for(int i = 0 ; i < size<0,0>(acc); i++) {
        //     for(int j = 0 ; j < size<0,1>(acc); j++) {
        //         for(int k = 0 ; k < size<1>(acc); k++) {
        //             for(int l = 0 ; l < size<2>(acc); l++) {
        //                 print(acc((i,j),k, l));
        //                 printf(" ");
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        printf("tCsA i:\n");
        print_tensor(tCsA(_, _, i));
        print(tCsA(_, _, i).layout());
        printf("\ntCsB i:\n");
        print_tensor(tCsB(_, _, i));
        print(tCsB(_, _, i).layout());
        printf("\ncurrent acc_s Tensor:\n");
        print_tensor(acc);
        print(acc.layout());
        printf("\n---------------------------------------------\n");
    }
#endif
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
#ifdef DEBUG
    if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        // print(size<2>(tCrA));
        printf("\n---------------------------------------------\n");
        printf("--- In gemm RESULT ---\n");
        printf("acc_s iteration times: ");
        print(size<2>(tCrA));
        printf("\n");
        // printf("\n---------------------------------------------\n");
        // print(acc);
        // printf("\n---------------------------------------------\n");
        // print(acc.shape());
        // printf("\n---------------------------------------------\n");
        // print(acc.size());
        // printf("\n---------------------------------------------\n");
        // print(acc.stride());
        // printf("\n---------------------------------------------\n");
        // for(int i = 0 ; i < size<0,0>(acc); i++) {
        //     for(int j = 0 ; j < size<0,1>(acc); j++) {
        //         for(int k = 0 ; k < size<1>(acc); k++) {
        //             for(int l = 0 ; l < size<2>(acc); l++) {
        //                 print(acc((i,j),k, l));
        //                 printf(" ");
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        printf("acc_s Tensor:\n");
        print_tensor(acc);
        printf("acc_s Layout:\n");
        print(acc.layout());
        printf("\n---------------------------------------------\n");
    }
#endif
}


template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
#ifdef DEBUG
    if(cute::thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // printf("\n---------------------------------------------\n");
        // printf("--- In convert_type BEFORE ---\n");
        // printf("\n---------------------------------------------\n");
        // print(tensor);
        // printf("\n---------------------------------------------\n");
        // print(tensor.shape());
        // printf("\n---------------------------------------------\n");
        // print(tensor.size());
        // printf("\n---------------------------------------------\n");
        // print(tensor.stride());
        // printf("\n---------------------------------------------\n");
        // for(int i = 0 ; i < size<0,0>(tensor); i++) {
        //     for(int j = 0 ; j < size<0,1>(tensor); j++) {
        //         for(int k = 0 ; k < size<1>(tensor); k++) {
        //             for(int l = 0 ; l < size<2>(tensor); l++) {
        //                 print(tensor((i,j),k, l));
        //                 printf(" ");
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // printf("\n---------------------------------------------\n");
    }
#endif
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
#ifdef DEBUG
    if(cute::thread0() && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // printf("\n---------------------------------------------\n");
        // printf("--- In convert_type AFTER ---\n");
        // printf("\n---------------------------------------------\n");
        // for (int i = 0 ; i < frag.size(); ++i) {
        //     print(frag[i]);
        //     printf(" ");
        // }
        // print(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()));
        // printf("\n---------------------------------------------\n");
        // print(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()).shape());
        // printf("\n---------------------------------------------\n");
        // print(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()).size());
        // printf("\n---------------------------------------------\n");
        // print(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()).stride());
        // printf("\n---------------------------------------------\n");
        // for(int i = 0 ; i < size<0,0>(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout())); i++) {
        //     for(int j = 0 ; j < size<0,1>(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout())); j++) {
        //         for(int k = 0 ; k < size<1>(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout())); k++) {
        //             for(int l = 0 ; l < size<2>(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout())); l++) {
        //                 print(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout())((i,j),k, l));
        //                 printf(" ");
        //             }
        //             printf("\n");
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        printf("\n---------------------------------------------\n");
    }
#endif
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    // Important: the order of the get<>(layout) CANNOT be changed!!!
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));         // TODO: ((2, MMA_M), (2, MMA_N))  
};



template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

} // End of namespace flash