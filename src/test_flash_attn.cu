#include "flash_fwd_hdim128_fp16_sm80.cu"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

// Helper function to initialize tensor data with sequential numbers
template<typename T>
void initialize_tensor(T* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>((unsigned int)i);  // Fill with 1, 2, 3, 4, ...
    }
}

// Helper function to zero initialize tensor data
template<typename T>
void zero_initialize_tensor(T* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(0.0f);
    }
}

// Helper function to check CUDA errors
// #define CHECK_CUDA(call) do {                                 \
//     cudaError_t err = call;                                  \
//     if (err != cudaSuccess) {                               \
//         printf("CUDA error at %s %d: %s\n", __FILE__,       \
//                __LINE__, cudaGetErrorString(err));          \
//         exit(EXIT_FAILURE);                                 \
//     }                                                       \
// } while(0)

void compute_qk_cpu(
    const cutlass::half_t* q,  // [batch_size, num_heads, seq_len, head_dim]
    const cutlass::half_t* k,  // [batch_size, num_heads, seq_len, head_dim]
    float* output,             // [batch_size, num_heads, seq_len, seq_len]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Loop over batches and heads
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Calculate base offsets for this batch and head
            size_t q_batch_offset = b * (num_heads * seq_len * head_dim);
            size_t k_batch_offset = b * (num_heads * seq_len * head_dim);
            size_t q_head_offset = h * (seq_len * head_dim);
            size_t k_head_offset = h * (seq_len * head_dim);
            size_t o_offset = b * (num_heads * seq_len * seq_len) + h * (seq_len * seq_len);

            // Compute Q * K^T for this batch and head
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        // Calculate indices for Q and K
                        size_t q_idx = q_batch_offset + q_head_offset + i * head_dim + d;
                        size_t k_idx = k_batch_offset + k_head_offset + j * head_dim + d;
                        
                        float q_val = static_cast<float>(q[q_idx]);
                        float k_val = static_cast<float>(k[k_idx]);
                        sum += q_val * k_val;
                    }
                    // Just store the dot product without scaling
                    output[o_offset + i * seq_len + j] = sum;
                }
            }
        }
    }
}

// Modify the comparison function
float compare_results(const cutlass::half_t* gpu_output, const float* cpu_output, 
                     size_t size, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    int diff_count = 0;
    
    for (size_t i = 0; i < size; i++) {
        float gpu_val = static_cast<float>(gpu_output[i]);
        float diff = std::abs(gpu_val - cpu_output[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 0.01f) {  // Count differences larger than 0.01
            diff_count++;
        }
    }
    
    std::cout << "Number of elements with difference > 0.01: " << diff_count 
              << " out of " << size << " elements (" 
              << (100.0f * diff_count / size) << "%)" << std::endl;
    
    return max_diff;
}

int main() {
    // Define problem dimensions
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 64;
    const int head_dim = 128;
    
    // Calculate sizes
    const size_t qk_size = batch_size * num_heads * seq_len * head_dim;
    const size_t o_size = batch_size * num_heads * seq_len * seq_len;
    
    // Allocate host memory
    cutlass::half_t *h_q = new cutlass::half_t[qk_size];
    cutlass::half_t *h_k = new cutlass::half_t[qk_size];
    cutlass::half_t *h_o = new cutlass::half_t[o_size];
    
    // Initialize input tensors
    initialize_tensor(h_q, qk_size);
    initialize_tensor(h_k, qk_size);
    // zero_initialize_tensor(h_o, o_size);
    initialize_tensor(h_o, o_size);
    
    // Allocate device memory
    cutlass::half_t *d_q, *d_k, *d_o;
    CHECK_CUDA(cudaMalloc(&d_q, qk_size * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_k, qk_size * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_o, o_size * sizeof(cutlass::half_t)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_q, h_q, qk_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, qk_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    // index o
    CHECK_CUDA(cudaMemcpy(d_o, h_o, o_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    
    // Initialize Flash Attention parameters
    Flash_fwd_params params;
    params.q_ptr = d_q;
    params.k_ptr = d_k;
    params.o_ptr = d_o;
    
    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seq_len;
    params.seqlen_k = seq_len;
    params.d = head_dim;
    params.seqlen_q_rounded = seq_len;
    params.seqlen_k_rounded = seq_len;
    params.d_rounded = head_dim;
    
    // Set strides
    params.q_batch_stride = num_heads * seq_len * head_dim;
    params.k_batch_stride = num_heads * seq_len * head_dim;
    params.o_batch_stride = num_heads * seq_len * seq_len;
    params.q_head_stride = seq_len * head_dim;
    params.k_head_stride = seq_len * head_dim;
    params.o_head_stride = seq_len * seq_len;
    params.q_row_stride = head_dim;
    params.k_row_stride = head_dim;
    params.o_row_stride = seq_len;
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Run Flash Attention
    run_mha_fwd_<cutlass::half_t, 128, false>(params, stream);
    
    // Wait for completion
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_o, d_o, o_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    
    // Print first few elements of output
    // std::cout << "\nInput Q tensor (first 10 elements):" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << static_cast<float>(h_q[i]) << " ";
    // }
    // std::cout << "\n\nInput K tensor (first 10 elements):" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //     std::cout << static_cast<float>(h_k[i]) << " ";
    // }
    std::cout << "\n\nOutput O tensor (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << static_cast<float>(h_o[i]) << " ";
    }
    std::cout << std::endl;

    // Compute CPU reference implementation
    std::vector<float> cpu_output(o_size);
    compute_qk_cpu(h_q, h_k, cpu_output.data(), 
                  batch_size, num_heads, seq_len, head_dim);
    
    // Compare results
    float max_diff = compare_results(h_o, cpu_output.data(), o_size);
    
    // Print comparison results
    std::cout << "\nCPU QK Output (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << cpu_output[i] << " ";
    }
    std::cout << "\n\nMaximum difference between CPU and GPU results: " << max_diff << std::endl;
    
    if (max_diff > 1e-3f) {
        std::cout << "WARNING: Results differ significantly!" << std::endl;
    } else {
        std::cout << "Results match within tolerance." << std::endl;
    }
    
    // Cleanup
    delete[] h_q;
    delete[] h_k;
    delete[] h_o;
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_o));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    return 0;
} 