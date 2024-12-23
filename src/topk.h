#pragma once


#include <cmath>

#include <cute/tensor.hpp>
#include <cuda_runtime.h>

#include <cutlass/numeric_types.h>

#include "utils.h"


namespace topk {

using namespace cute;

template<typename T>
struct __align__(4) Node {
    T value;
    uint8_t index;
    __forceinline__ __device__ Node() : value(0), index(0) {};
    __forceinline__ __device__ Node(T value, uint8_t index) : value(value), index(index) {};
};// Thr_tile Bitonic Sort

template<int kNRows, int kNCols, typename T>
struct TopK {
    Node<T> nodes[kNRows * kNCols];

    __forceinline__ __device__ TopK() {}


    // Thr_tile Bitonic Sort
}
    
}

