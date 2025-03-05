cmake_minimum_required(VERSION 3.18)
project(flash_attention CUDA CXX)

# Set CUDA architecture - making this more explicit
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set(CMAKE_CXX_STANDARD 17)          # 指定 C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 必须使用指定版本

set(CMAKE_PREFIX_PATH /usr/include/libtorch)
find_package(Torch REQUIRED)

# Find Python with all required components
# find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(Torch REQUIRED NO_MODULE PATHS /usr/include/libtorch NO_DEFAULT_PATH)
# find_package(ATen REQUIRED NO_MODULE PATHS /usr/include/libtorch/share/cmake/ATen NO_DEFAULT_PATH)
enable_language(CUDA)

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# include_directories(${Python_INCLUDE_DIRS})

# Create separate executables for each source file
add_executable(torch_api_test src/torch_api_test.cu)
add_executable(test_flash_attn src/test_flash_attn.cu)

# Set CUDA specific properties for both targets
foreach(target torch_api_test test_flash_attn)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80"
    )

    # Include directories for each target
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${INCLUDE_DIRS}
        # ${Python_INCLUDE_DIRS}
    )

    # Link libraries for each target
    target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})

    # Set compiler flags for each target
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --extended-lambda
            -O2
            -std=c++17
            -arch=sm_80
            -g
            -G
        >
    )
endforeach()
