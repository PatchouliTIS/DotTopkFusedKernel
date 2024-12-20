cmake_minimum_required(VERSION 3.18)
project(flash_attention CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CXX_STANDARD 17)          # 指定 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 必须使用指定版本

# Find Python with all required components
# find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(Torch REQUIRED NO_MODULE PATHS /usr/include/libtorch NO_DEFAULT_PATH)
# find_package(ATen REQUIRED NO_MODULE PATHS /usr/include/libtorch/share/cmake/ATen NO_DEFAULT_PATH)
enable_language(CUDA)

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# include_directories(${Python_INCLUDE_DIRS})

# Set source files
set(SOURCES
    src/test_flash_attn.cu
)

# Create executable
add_executable(test_flash_attn ${SOURCES})
# add_definitions("-g -G")

# Set CUDA specific flags
set_target_properties(test_flash_attn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80"
)

# Include directories
target_include_directories(test_flash_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${INCLUDE_DIRS}
    # ${Python_INCLUDE_DIRS}
)

# target_link_libraries(test_flash_attn ${TORCH_LIBRARIES} Python::Python)

# Set compiler flags
target_compile_options(test_flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -O2
        -std=c++17
    >
)
