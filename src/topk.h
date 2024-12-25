#pragma once


#include <cmath>

#include <cute/tensor.hpp>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include "utils.h"

// epsilon
#define FLOAT_EPSILON 1e-6

// max
#define MAX_VALUE(a, b) ((a) > (b) ? (a) : (b))

// float equal
#define IS_FLOAT_EQUAL(a, b) ((a) - (b) < 0 ? (b) - (a) <= FLOAT_EPSILON : (a) - (b) <= FLOAT_EPSILON)

namespace topk {

using namespace cute;

// template<typename T>
// struct __align__(8) Node {
//     T value;
//     uint8_t index;
    
//     __forceinline__ __device__ Node() : value(0), index(0) {};
//     __forceinline__ __device__ Node(T value, uint8_t index) : value(value), index(index) {};
    
//     __device__ bool operator<(const Node& other) const {
//         return value < other.value;
//     }
    
//     __device__ static Node warp_reduce_max(Node val, unsigned mask = 0xffffffff) {
//         #pragma unroll
//         for (int offset = 16; offset > 0; offset /= 2) {
//             T other_val = __shfl_down_sync(mask, val.value, offset);
//             uint8_t other_idx = __shfl_down_sync(mask, val.index, offset);
            
//             if (other_val > val.value) {
//                 val.value = other_val;
//                 val.index = other_idx;
//             }
//         }
//         return val;
//     }
    
    
// };// Thr_tile Bitonic Sort




template<int kNRows, int kNCols, typename T>
struct TopK {
private:
    uint16_t index[kNRows * kNCols];

    __forceinline__ __device__ TopK() {}

    // Thr_tile Bitonic Sort

    __forceinline__ __device__  void warp_reduce_pairs(T& val, uint16_t &idx, const unsigned int &tid, unsigned mask = 0xffffffff) {
        const uint32_t lane_id = tid & 0x1f;
        T max_val = MAX_VALUE(val, __shfl_down_sync(mask, val, 1, 2));
        uint16_t other_idx = __shfl_down_sync(mask, idx, 1, 2);

        if ((lane_id & 1) == 0) {
            // exchange happens
            if(!IS_FLOAT_EQUAL(max_val, val)) {
                val = max_val;
                // get the index of the bigger value
                idx = other_idx; 
            }
        }
        return ;
    }

    __forceinline__ __device__ static void warp_reduce_quads(T& val, uint16_t &idx, const unsigned int &tid, unsigned mask = 0xffffffff) {
        const uint32_t lane_id = tid & 0x1f;
        T max_val = MAX_VALUE(val, __shfl_down_sync(mask, val, 2, 4));
        uint16_t other_idx = __shfl_down_sync(mask, idx, 2, 4);

        if ((lane_id & 0x3) == 0) {
            if(!IS_FLOAT_EQUAL(max_val, val)) {
                val = max_val;
                idx = other_idx;
            }
        }
        return ;
    }


public:


    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void topk_index(Tensor0 &acc_s) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        static_assert(decltype(size<1>(scores))::value == kNCols);

        // Step1: Bitonic Sort within current thread
        // Step2: Warp-shuffle pair-wise(2 stride) to get 16 topk values and index in every 32 valuse
        // Step3: Bitonic Sort within thread (tid & 0x1 == 0)
        // Step4: Warp-shuffle quad-wise(4 stride) to get 16 topk values and index in every 32 valuse
        // Step5: Bitonic Sort within thread (tid & 0x3 == 0), get final topk values and index in current global k-th block
        // Step6: Update global topk index in reg memory
    }
    


}
    
}

