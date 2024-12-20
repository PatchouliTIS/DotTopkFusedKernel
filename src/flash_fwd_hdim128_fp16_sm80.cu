#include "kernel_launch_template.h"


// The function needs to be declared as a template first
template<typename T, int HEADDIM, bool USE_DROPOUT>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template<>
void run_mha_fwd_<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, false>(params, stream);
}