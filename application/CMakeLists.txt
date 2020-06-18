cmake_minimum_required(VERSION 3.8)

project(application VERSION 0.0.1)

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

add_executable(test_mmul test/mmul.cpp)
target_include_directories(test_mmul INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_mmul executor CONAN_PKG::doctest Eigen3::Eigen ${MMUL_CUDA})
set_target_properties(test_mmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

doctest_discover_tests(
        test_mmul
        EXTRA_ARGS
        -s)
