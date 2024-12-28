#include "flash_fwd_hdim128_fp16_sm80.cu"
#include <cuda_runtime.h>
// #include <torch/torch.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>



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
// Helper function to initialize tensor data with sequential numbers
template<typename T>
void initialize_tensor_float(T* data, size_t size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // Create uniform distribution between -1 and 1
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; i++) {
        // Generate a unique value by combining index and random number
        float random_part = distribution(gen);
        // Combine index and random part to ensure uniqueness
        // Scale down the index to keep values in reasonable range
        float value = random_part + (static_cast<float>(i) / size) * 0.001f;
        // float value = float(i) * 0.1f;
        data[i] = value;
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
    uint16_t* indices,         // [batch_size, num_heads, seq_len, topk]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int topk
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
            size_t idx_offset = b * (num_heads * seq_len * topk) + h * (seq_len * topk);

            // Compute Q * K^T for this batch and head
            for (int i = 0; i < seq_len; i++) {
                // Store dot products and indices for sorting
                std::vector<std::pair<float, uint16_t>> row_values;
                
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        size_t q_idx = q_batch_offset + q_head_offset + i * head_dim + d;
                        size_t k_idx = k_batch_offset + k_head_offset + j * head_dim + d;
                        
                        float q_val = static_cast<float>(q[q_idx]);
                        float k_val = static_cast<float>(k[k_idx]);
                        sum += q_val * k_val;
                    }
                    output[o_offset + i * seq_len + j] = sum;
                    row_values.push_back({sum, static_cast<uint16_t>(j)});
                }

                // Sort row values in descending order
                std::sort(row_values.begin(), row_values.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // Store top-k indices
                for (int t = 0; t < topk; t++) {
                    indices[idx_offset + i * topk + t] = row_values[t].second;
                }
            }
        }
    }
}



int main() {
    // Define problem dimensions
    const int batch_size = 1;
    const int num_heads = 1;
    const int seq_len = 256;
    const int head_dim = 128;
    const int topk = 16;
    
    // Calculate sizes
    const size_t qk_size = batch_size * num_heads * seq_len * head_dim;
    const size_t o_size = batch_size * num_heads * seq_len * seq_len;
    const size_t topk_size = batch_size * num_heads * seq_len * topk;
    
    // Allocate host memory
    cutlass::half_t *h_q = new cutlass::half_t[qk_size];
    cutlass::half_t *h_k = new cutlass::half_t[qk_size];
    uint16_t *h_ido = new uint16_t[topk_size];
    cutlass::half_t *h_o = new cutlass::half_t[o_size]; 

    
    
    
    // Initialize input tensors
    initialize_tensor_float(h_q, qk_size);
    initialize_tensor_float(h_k, qk_size);
    zero_initialize_tensor(h_o, o_size);
    zero_initialize_tensor(h_ido, topk_size);

    // Allocate device memory
    cutlass::half_t *d_q, *d_k,*d_o;
    uint16_t *d_ido;
    CHECK_CUDA(cudaMalloc(&d_q, qk_size * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_k, qk_size * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_o, o_size * sizeof(cutlass::half_t)));
    CHECK_CUDA(cudaMalloc(&d_ido, topk_size * sizeof(uint16_t)));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_q, h_q, qk_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, qk_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    // TEST: index o
    // CHECK_CUDA(cudaMemcpy(d_o, h_o, o_size * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
    
    // Initialize Flash Attention parameters
    Flash_fwd_params params;
    params.q_ptr = d_q;
    params.k_ptr = d_k;
    params.o_ptr = d_o;
    params.ido_ptr = d_ido;
    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.seqlen_q = seq_len;
    params.seqlen_k = seq_len;
    params.d = head_dim;
    params.seqlen_q_rounded = seq_len;
    params.seqlen_k_rounded = seq_len;
    params.d_rounded = head_dim;
    params.topk = topk;
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
    params.ido_batch_stride = num_heads * seq_len * topk;
    params.ido_head_stride = seq_len * topk;
    params.ido_row_stride = topk;
    
    // Run Flash Attention
    run_mha_fwd_<cutlass::half_t, 128, false>(params, stream);
    
    // Wait for completion
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_o, d_o, o_size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ido, d_ido, topk_size * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    
    std::cout << "\n\nOutput IDO tensor:" << std::endl;
    for (int b = 0; b < batch_size; b++) {
        std::cout << "Batch " << b << ":\n";
        for (int h = 0; h < num_heads; h++) {
            std::cout << "  Head " << h << ":\n";
            for (int s = 0; s < seq_len; s++) {
                std::cout << "    Seq " << s << ": ";
                for (int t = 0; t < topk; t++) {
                    size_t idx = b * params.ido_batch_stride + 
                                h * params.ido_head_stride +
                                s * params.ido_row_stride + t;
                    std::cout << h_ido[idx] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;


    // Compute CPU reference implementation
    std::vector<float> cpu_output(o_size);
    std::vector<uint16_t> cpu_indices(topk_size);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    compute_qk_cpu(h_q, h_k, cpu_output.data(), cpu_indices.data(),
                  batch_size, num_heads, seq_len, head_dim, topk);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    // Print timing results
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "CPU Execution time: " << cpu_duration.count() << " ms" << std::endl;
    // std::cout << "Speedup: " << static_cast<float>(cpu_duration.count()) / gpu_milliseconds << "x" << std::endl;
    
    // Compare QK results
    float max_diff = compare_results(h_o, cpu_output.data(), o_size);
    
    // Compare indices
    std::cout << "\nComparing indices between CPU and GPU:\n";
    std::cout << "CPU indices:\n";
    for (int b = 0; b < batch_size; b++) {
        std::cout << "Batch " << b << ":\n";
        for (int h = 0; h < num_heads; h++) {
            std::cout << "  Head " << h << ":\n";
            for (int s = 0; s < seq_len; s++) {
                std::cout << "    Seq " << s << ": ";
                for (int t = 0; t < topk; t++) {
                    size_t idx = b * params.ido_batch_stride + 
                                h * params.ido_head_stride +
                                s * params.ido_row_stride + t;
                    std::cout << cpu_indices[idx] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    // Cleanup
    delete[] h_q;
    delete[] h_k;
    // delete[] h_o;
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    // CHECK_CUDA(cudaFree(d_o));
    CHECK_CUDA(cudaFree(d_ido));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    return 0;
} 