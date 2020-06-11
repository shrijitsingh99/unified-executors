cmake_minimum_required(VERSION 3.8)

project(application VERSION 0.0.1)

find_package(doctest REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE REQUIRED)
find_package(CUDA QUIET)

include_directories(include)

if(CUDA_FOUND)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    enable_language(CUDA)

    add_definitions(-DCUDA)

    add_library(mmul_cuda STATIC src/mmul.cu)
    set(MMUL_CUDA mmul_cuda)
endif()

add_executable(mmul_test test/mmul.cpp)
target_include_directories(mmul_test INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mmul_test executor doctest::doctest Eigen3::Eigen ${MMUL_CUDA})
set_target_properties(mmul_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

doctest_discover_tests(
        mmul_test
        TEST_PREFIX
        "mmul.")