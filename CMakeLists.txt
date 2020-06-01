cmake_minimum_required(VERSION 3.8)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(unified_api_experiments VERSION 0.0.1 LANGUAGES CXX CUDA)

find_package(CUDA)

add_subdirectory(unified_executor)

include_directories(include)

if(CUDA_FOUND)
    add_definitions(-DCUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    add_library(mmul_cuda STATIC src/mmul.cu)
endif()

add_executable(executor_example src/executor.cpp)

target_link_libraries(executor_example unified_api_executor mmul_cuda)
set_target_properties(executor_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
