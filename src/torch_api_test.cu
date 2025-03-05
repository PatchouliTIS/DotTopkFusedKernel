#include <torch/torch.h>
#include <chrono>

// using namespace torch;

void torch_api_test(
    const int& batch_size,
    const int& num_heads,
    const int& seq_len,
    const int& head_dim
) {
    torch::TensorOptions options =  torch::TensorOptions()
        .dtype( torch::kHalf)
        .device( torch::kCUDA, 0);
    auto q = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    auto k = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    
    for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            for (int64_t s = 0; s < seq_len; s++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    // Example initialization: combine indices to create unique values
                    size_t i = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim + d;
                    at::Half value = static_cast<at::Half>(0.01f * (i % 128) + 0.0001f * i);
                    
                    q[b][h][s][d] = value;
                    k[b][h][s][d] = value;
                }
            }
        }
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    float milliseconds = 0;

    // Time matmul operation
    cudaEventRecord(start);
    cudaEventRecord(total_start);
    torch::Tensor C = torch::matmul(q, k.transpose(2, 3));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Matrix multiplication time: " << milliseconds << " ms" << std::endl;

    // Time topk operation
    cudaEventRecord(start);
    auto topk_out = torch::topk(C, 16, /*dim=*/-1);
    cudaEventRecord(stop);
    cudaEventRecord(total_stop);
    cudaEventSynchronize(stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Top-k operation time: " << milliseconds << " ms" << std::endl;
    cudaEventElapsedTime(&milliseconds, total_start, total_stop);
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;

    auto values = std::get<0>(topk_out);  // top-16 values
    auto indices = std::get<1>(topk_out); // corresponding indices

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print original matrix multiplication result
    std::cout << "Result of q * k:\n" << C << std::endl;
    
    // Print top-k results
    std::cout << "Top 16 values:\n" << values << std::endl;
    std::cout << "Top 16 indices:\n" << indices << std::endl;

    // 4. 打印相关信息
    std::cout << "Matrix q (FP16 on GPU):\n" << q << std::endl;
    std::cout << "Matrix k (FP16 on GPU):\n" << k << std::endl;
    std::cout << "C.dtype()  = " << C.dtype() << std::endl;
    std::cout << "C.device() = " << C.device() << std::endl;
}

int main() {
    std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    
    torch_api_test(1, 16, 1024, 128);
    return 0;
}