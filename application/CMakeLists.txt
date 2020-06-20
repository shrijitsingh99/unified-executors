cmake_minimum_required(VERSION 3.8)

project(application VERSION 0.0.1)

find_package (Eigen3 3.3 REQUIRED)
find_package(CUDA QUIET)

include_directories(include)

if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    add_definitions(-DCUDA)
endif()

add_executable(test_mmul test/mmul.cpp src/mmul.cu)
target_include_directories(test_mmul INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_mmul executor CONAN_PKG::doctest Eigen3::Eigen ${MMUL_CUDA})

doctest_discover_tests(
        test_mmul
        EXTRA_ARGS
        -s)
