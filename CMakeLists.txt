# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.18)
project(AtA-gpu CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures for M1/M2 Macs
set(CMAKE_CUDA_ARCHITECTURES 87)  # For Apple Silicon GPUs

# Find required packages
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Add library target
add_library(ata_gpu STATIC
    src/rtxx.cpp
    src/fused_kernels.cu
)

# Set CUDA specific properties
set_target_properties(ata_gpu PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link against CUDA libraries
target_link_libraries(ata_gpu PRIVATE 
    CUDA::cudart
    CUDA::cublas
)

# Add executable target (if you have a main file)
add_executable(ata_test src/main.cpp)
target_link_libraries(ata_test PRIVATE ata_gpu)

# Installation rules
install(TARGETS ata_gpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    src/rtxx.h
    DESTINATION include
)
