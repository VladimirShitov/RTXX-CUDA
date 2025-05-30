# https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AtA-gpu LANGUAGES CXX)

find_package(CUDA REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED YES)

add_executable(ata-sp ${CMAKE_SOURCE_DIR}/src/ata.cpp)
target_link_libraries(ata-sp ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

add_executable(ata-dp ${CMAKE_SOURCE_DIR}/src/ata.cpp)
target_link_libraries(ata-dp ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_compile_definitions(ata-dp PUBLIC -DFLOAT_AS_DOUBLE)

if(APPLE)
  set_property(TARGET ata-sp ata-dp PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)
