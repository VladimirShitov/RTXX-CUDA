# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.18)
project(AtA-gpu CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Configuration
if(APPLE)
    # For Apple Silicon GPUs
    set(CMAKE_CUDA_ARCHITECTURES 87)
else()
    # For other systems, detect architecture or set manually
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Enable CUDA language
enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Print CUDA information
message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA Library Dirs: ${CUDAToolkit_LIBRARY_DIR}")

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
target_link_libraries(ata_gpu PUBLIC 
    CUDA::cudart
    CUDA::cublas
)

# Add executable target
add_executable(ata src/ata.cpp)
target_link_libraries(ata PRIVATE ata_gpu)

# Installation rules
install(TARGETS ata_gpu
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    src/rtxx.h
    DESTINATION include
)
