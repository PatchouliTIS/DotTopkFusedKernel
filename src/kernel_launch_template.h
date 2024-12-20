#pragma once

#include "kernel_params.h"
#include "kernel_traits.h"
#include <cuda_runtime.h>
#include "kernel_fwd.h"
// #include "c10/cuda/CUDAException.h"
// #include "c10/cuda/CUDAGuard.h"
// #include "c10/cuda/CUDAStream.h"
// #include "ATen/cuda/CUDAGeneratorImpl.h"
#include "hardware_info.h"


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()


// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// __global__ void printARCH() {
//     printf("ARCH_SUPPORTS_FLASH: %d\n", __CUDA_ARCH__);
// }
// printARCH<<<1, 1>>>();
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
// #else
// #define KERNEL_PARAM_MODIFIER
// #endif




#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)


DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        flash::compute_attn<Kernel_traits, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
};

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");


template<typename Kernel_traits>
void run_fused_mtkernel(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // printARCH<<<1, 1>>>();
    
    const bool Is_even_MN =  params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool Is_even_K = params.d == Kernel_traits::kHeadDim;

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    BOOL_SWITCH(Is_even_MN, Is_even_MNConst, [&] {
        BOOL_SWITCH(Is_even_K, Is_even_KConst, [&] {
            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_even_MNConst && Is_even_KConst, Is_even_KConst>;
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            // C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}




template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    // run_fused_mtkernel<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>>(params, stream);
    run_fused_mtkernel<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>>(params, stream);
            
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    
    // // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
    // // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
    // // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
    // // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // // 1st ones are good for H100, A100
    // // 2nd one is good for A6000 bc we get slightly better occupancy
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
    // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
}