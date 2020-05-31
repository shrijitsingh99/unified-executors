cmake_minimum_required(VERSION 3.8)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(unified_api_experiments VERSION 0.0.1 LANGUAGES CXX CUDA)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(CUDA)

add_definitions(-DCUDA)

include_directories("${CUDA_INCLUDE_DIRS}")


include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
add_executable(template_meta_plg src/template_meta_plg.cpp)
add_executable(executor src/executor.cpp include/execution/executor.cu src/mmul.cu)

target_link_libraries(executor Eigen3::Eigen OpenMP::OpenMP_CXX)
set_target_properties(executor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
