# CUDA Flash Attention with TopK Implementation

A CUDA implementation of Flash Attention with TopK selection, built using CUTLASS and CUTE (CUDA Templates) libraries.

## Overview

This project implements an efficient Flash Attention mechanism with TopK selection on NVIDIA GPUs. It uses CUTLASS for efficient matrix operations and CUTE for tensor abstractions.

## Features

- Flash Attention implementation optimized for NVIDIA GPUs
- TopK selection integrated into attention computation
- Support for FP16 data type
- Configurable batch size, number of heads, sequence length, and head dimensions
- CPU reference implementation for result verification

## Requirements

- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- NVIDIA GPU with Compute Capability 8.0 or higher (Ampere architecture)
- C++17 compiler

## Build Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Build the project:
```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Run the test program:
```bash
./test_flash_attn
```

This will execute the Flash Attention kernel and print the results.

## Project Structure

- `src/`
  - `test_flash_attn.cu`: Main test file and CPU reference implementation
  - `kernel_fwd.h`: Forward pass kernel implementation
  - `kernel_traits.h`: Kernel configuration traits
  - `topk.h`: TopK implementation
  - `utils.h`: Utility functions and helpers
- `include/`: External dependencies and headers

## Implementation Details

- Uses CUTLASS's MMA (Matrix Multiply-Accumulate) operations for efficient attention computation
- Implements custom TopK selection using bitonic sort
- Supports configurable block sizes and thread organization
- Includes debug printing capabilities for development

## Performance Considerations

- Optimized for Ampere architecture (SM80)
- Uses shared memory for efficient data access
- Implements efficient memory access patterns
- Utilizes Tensor Core matrix operations

## Acknowledgments

- NVIDIA CUTLASS library
- CUDA Templates (CUTE) library