#pragma once



#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include "cutlass/fast_math.h"

#include <cutlass/numeric_types.h>

#include "utils.h"

// epsilon
#define FLOAT_EPSILON 1e-6

// max
// #define MAX_VALUE(a, b) ((a) > (b) ? (a) : (b))

// float equal
// #define IS_FLOAT_EQUAL(a, b) ((a) - (b) < 0 ? (b) - (a) <= FLOAT_EPSILON : (a) - (b) <= FLOAT_EPSILON)

#define ORDERV(x,idx,row,a,b) { bool swap = reverse ^ (x(row,a)<x(row,b)); \
      float auxa = x(row,a); \
      uint16_t auxidx = idx(row,a); \
      if (swap) { x(row,a) = x(row,b); x(row,b) = auxa; idx(row,a) = idx(row,b); idx(row,b) = auxidx; } }

#define B2V(x,idx,row,a) { ORDERV(x,idx,row,a,a+1) }
#define B4V(x,idx,row,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,idx,row,a+i4,a+i4+2) } B2V(x,idx,row,a) B2V(x,idx,row,a+2) }
#define B8V(x,idx,row,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,idx,row,a+i8,a+i8+4) } B4V(x,idx,row,a) B4V(x,idx,row,a+4) }
#define B16V(x,idx,row,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,idx,row,a+i16,a+i16+8) } B8V(x,idx,row,a) B8V(x,idx,row,a+8) }
#define B32V(x,idx,row,a) { for (int i32=0;i32<16;i32++) { ORDERV(x,idx,row,a+i32,a+i32+16) } B16V(x,idx,row,a) B16V(x,idx,row,a+16) }
#define B64V(x,idx,row,a) { for (int i64=0;i64<32;i64++) { ORDERV(x,idx,row,a+i64,a+i64+32) } B32V(x,idx,row,a) B32V(x,idx,row,a+32) }

namespace flash {

using namespace cute;

template<int kNRows, int kNCols, int strideBitInThr, int strideBitAmongThr, int Block_N, typename T, typename IDX>
struct TopK {
public:
    // using TensorIndex = decltype(make_tensor<IDX>(Shape<Int<kNRows>, Int<kNCols>>{}));
    // using TensorValue = decltype(make_tensor<T>(Shape<Int<kNRows>, Int<kNCols>>{}));
    // TensorIndex global_index;
    // TensorValue global_value;

    __forceinline__ __device__ TopK() {}

    // Thr_tile Bitonic Sort
    template<typename Operator>
    __forceinline__ __device__ void warp_reduce_pairs(float& val, IDX& idx, Operator& Op, unsigned mask = 0xffffffff) {
        float other_val = __shfl_down_sync(mask, val, 1, 2);
        float max_val = Op(val, other_val);
        IDX other_idx = __shfl_down_sync(mask, idx, 1, 2);

        if(max_val > val) {
            val = max_val;
            idx = other_idx;
        }
    }

    template<typename Operator>
    __forceinline__ __device__ void warp_reduce_quads(float& val, IDX& idx, Operator& Op, const unsigned int& tid, unsigned mask = 0xffffffff) {
        const uint32_t lane_id = tid & 0x1f;
        float other_val = __shfl_down_sync(mask, val, 2, 4);
        float max_val = Op(val, other_val);
        IDX other_idx = __shfl_down_sync(mask, idx, 2, 4);

        if ((lane_id & 0x3) == 0) {
            if(max_val > val) {
                val = max_val;
                idx = other_idx;
            }
        }
    }
    
    template<bool Is_first, typename index_t, typename Tensor0, typename Tensor1, typename Tensor2>
    __forceinline__ __device__ void topk_index(Tensor0 &acc_s, Tensor1 &global_index, Tensor2 &global_value, const unsigned int &tid, const int &n_idx) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        flash::MaxOp<float> max_op;
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
        static_assert(decltype(size<1>(scores))::value == kNCols);
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("Entering topk_index\n");
            printf("tid: %d\n", tid);
            printf("n_idx: %d\n", n_idx);
            printf("Block_N: %d\n", Block_N);
            printf("kNRows: %d\n", kNRows);
            printf("kNCols: %d\n", kNCols);
            printf("---------------------THREAD 2 scores----------------------\n");
            print_tensor(scores);
            printf("\n---------------------THREAD 2 acc_s----------------------\n");
            print_tensor(acc_s);
        }
#endif
        // Step 0: Init index
        Tensor cur_idx = make_tensor_like<index_t>(scores);
        // The number of elements between two threads within one row are defined by layout<0,0>TiledMMA.Layout_C / Atom_MMA_M
        //                                                                                                  32   /   16
        IDX offset_blk = n_idx * Block_N;
        IDX offset_thr = ((tid & 0x3) << strideBitAmongThr);
        #pragma unroll
        for(int row = 0; row < kNRows; row++) {
            #pragma unroll
            for(int col = 0; col < kNCols; col++) {
                // take thread 0 for example
                // coord in acc_s(reg): 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
                // index in tSgS(gmem): 0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57
                // the layout of TV are defined in TiledMMA Layout_C since the acc_s is tiled from tSgS by TiledMMA
                cur_idx(row, col) = ((col >> 1) << strideBitInThr) + ((col & 0x1)) + offset_thr + offset_blk;
            }
        }

#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 0 RESULT cur_idx ---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n ---- scores ----\n");
            print_tensor(scores);
            printf("\n---------------------------------------------\n");
        }

        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 3) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 0 RESULT cur_idx THREAD 3---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n ---- scores ----\n");
            print_tensor(scores);
            printf("\n---------------------------------------------\n");
        }
#endif
        

        // Step1: Bitonic Sort within current thread
        // recursive in plane
        bool reverse;
        #pragma unroll
        for(int row = 0; row < kNRows; row++) {
            // 2-stride bitonic sort
            #pragma unroll
            for (int i=0; i< kNCols; i+=2) {
                reverse = ((i >> 1) + 1)&1;
                B2V(scores, cur_idx, row, i);
            }
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK Step 1 B2V RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            printf("scores: \n");
            print_tensor(scores);
            printf("cur_idx: \n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif
            // 4-stride bitonic sort
            #pragma unroll
            for (int i=0; i< kNCols; i+=4) {
                reverse = ((i >> 2) + 1)&1;
                B4V(scores, cur_idx, row, i);
            }
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK Step 1 B4V RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            printf("scores: \n");
            print_tensor(scores);
            printf("cur_idx: \n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif
            // 8-stride bitonic sort
            #pragma unroll
            for (int i=0; i< kNCols; i+=8) {
                reverse = ((i >> 3) + 1)&1;
                B8V(scores, cur_idx, row, i);
            }
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK Step 1 B8V RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            printf("scores: \n");
            print_tensor(scores);
            printf("cur_idx: \n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif
            // final 16 elements bitonic sort, reverse according to current  thread id
            reverse = (tid + 1) & 0x1;
            B16V(scores, cur_idx, row, 0);
        }

#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 1 RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }

        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 3) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 1 RESULT cur_idx THREAD 3---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif
        
        // Step2: Warp-shuffle pair-wise(2 stride) to get 16 topk values and index in every 32 valuse
        // Collect every two adjacent thread's value to the one having minor thread idx
        #pragma unroll
        for(int row = 0; row < kNRows; row++) {
            #pragma unroll
            for(int col = 0; col < kNCols; col++) {
                warp_reduce_pairs(scores(row, col), cur_idx(row, col), max_op, 0xffffffff);
            }
        }

#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 2 RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }

        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 3) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 2 RESULT cur_idx THREAD 3---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif

        // Step3: Bitonic Sort within thread (tid & 0x1 == 0)
        if((tid & 0x1) == 0) {
            #pragma unroll
            for(int row = 0; row < kNRows; row++) {
                // 2-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=2) {
                    reverse = ((i >> 1) + 1)&1;
                    B2V(scores, cur_idx, row, i);
                }
                // 4-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=4) {
                    reverse = ((i >> 2) + 1)&1;
                    B4V(scores, cur_idx, row, i);
                }
                // 8-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=8) {
                    reverse = ((i >> 3) + 1)&1;
                    B8V(scores, cur_idx, row, i);
                }
                // mid-term bitonic sort direction
                reverse = ((tid >> 1) + 1) & 0x1;
                B16V(scores, cur_idx, row, 0);
            }
        }
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 3 RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif


        // Step4: Warp-shuffle quad-wise(4 stride) to get 16 topk values and index in every 32 valuse
        // Collect the topk elements in the first thread of every 4 threads, the number 4 is presented as THREAD STRIDE, which are defined by shape<0,0>(TiledMMA.Layout_C) ((4,8),(2,2))
        #pragma unroll
        for(int row = 0; row < kNRows; row++) {
            #pragma unroll
            for(int col = 0; col < kNCols; col++) {
                warp_reduce_quads(scores(row, col), cur_idx(row, col), max_op, tid, 0xffffffff);
            }
        }// Now the topk 16 elements are in the first thread of each row, thread 0 for row 0, thread 4 for row 1, the thread stride are defined by shape<0,0>(TiledMMA.Layout_C) ((4,8),(2,2))
        
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 4 RESULT cur_idx ---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
            printf("Entering STEP 5\n");
            printf("tid & 0x3 == %d\n", tid & 0x3);
        }
#endif
        // Step5: Bitonic Sort within thread (tid & 0x3 == 0), get final topk values and index in current global k-th block
        // 0x3 mask are defined by shape<0,0>(TiledMMA.Layout_C) ((4,8),(2,2))
        if((tid & 0x3) == 0) {
#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("Entered STEP 5\n");
        }
#endif
            
            #pragma unroll
            for(int row = 0; row < kNRows; row++) {
                // 2-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=2) {
                    reverse = ((i >> 1) + 1)&1;
                    B2V(scores, cur_idx, row, i);
                }
                // 4-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=4) {
                    reverse = ((i >> 2) + 1)&1;
                    B4V(scores, cur_idx, row, i);
                }
                // 8-stride bitonic sort
                #pragma unroll
                for (int i=0; i< kNCols; i+=8) {
                    reverse = ((i >> 3) + 1)&1;
                    B8V(scores, cur_idx, row, i);
                }
                // Final bitonic sort direction, this value is according to whether current topk round is the first round.
                // If it is the first round, the bitonic sort direction is DECREMENTAL, otherwise it is INCREMENTAL and need to run another bitonic sort.
                reverse = !Is_first;
                B16V(scores, cur_idx, row, 0);
            }

#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 5 RESULT cur_idx THREAD 0---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }


        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 5 RESULT cur_idx THREAD 2---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(cur_idx);
            printf("\n---------------------------------------------\n");
        }
#endif
            

            // Step6: Update global topk index in reg memory (first or not)
            // Tensor acc_o = make_tensor(global_value.data(), flash::convert_layout_acc_rowcol(global_value.layout()));
            // Tensor idx_o = make_tensor(global_index.data(), flash::convert_layout_acc_rowcol(global_index.layout()));
            if(Is_first) {
                #pragma unroll
                for(int row = 0; row < kNRows; row++) {
                    #pragma unroll
                    for(int col = 0; col < kNCols; col++) {
                        global_index(row, col) = cur_idx(row, col);
                        global_value(row, col) = scores(row, col);
                    }
                }
            } else {   
                // Update the global topk index and value
                #pragma unroll
                for(int row = 0; row < kNRows; row++) {
                    #pragma unroll
                    for(int col = 0; col < kNCols; col++) {
                        if (max_op(global_value(row, col), scores(row, col)) > global_value(row, col)) {
                            global_index(row, col) = cur_idx(row, col);
                            global_value(row, col) = scores(row, col);
                        }
                    }
                }

                // Sort the global
                #pragma unroll
                for(int row = 0; row < kNRows; row++) {
                    // 2-stride bitonic sort
                    #pragma unroll
                    for (int i=0; i< kNCols; i+=2) {
                        reverse = ((i >> 1) + 1)&1;
                        B2V(global_value, global_index, row, i);
                    }
                    // 4-stride bitonic sort
                    #pragma unroll
                    for (int i=0; i< kNCols; i+=4) {
                        reverse = ((i >> 2) + 1)&1;
                        B4V(global_value, global_index, row, i);
                    }
                    // 8-stride bitonic sort
                    #pragma unroll
                    for (int i=0; i< kNCols; i+=8) {
                        reverse = ((i >> 3) + 1)&1;
                        B8V(global_value, global_index, row, i);
                    }
                    // final 16 elements bitonic sort, reverse according to current  thread id
                    reverse = false;
                    B16V(global_value, global_index, row, 0);
                }
            }


#ifdef DEBUG
        if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 2) {
            printf("\n---------------------------------------------\n");
            printf("--- In TopK STEP 6 RESULT global_index ---\n");
            printf("\n---------------------------------------------\n");
            print_tensor(global_index);
            printf("\n---------------------------------------------\n");
        }
#endif
        }
    }
    
};
    
};

