cmake_minimum_required(VERSION 3.8)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(unified_api_executor VERSION 0.0.1 LANGUAGES CXX CUDA)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(CUDA)

add_definitions(-D_SSE)

if(OpenMP_CXX_FOUND)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
endif()

if(CUDA_FOUND)
    add_definitions(-DCUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
endif()

include_directories(include)

add_library(unified_api_executor INTERFACE)
target_include_directories(unified_api_executor INTERFACE message("${PROJECT_SOURCE_DIR}/include"))
target_compile_features(unified_api_executor INTERFACE cxx_std_17)
target_link_libraries(unified_api_executor INTERFACE Eigen3::Eigen ${OpenMP_LIBS} ${CUDA_LIBRARIES})
