cmake_minimum_required(VERSION 3.5)

project(application VERSION 0.3)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package (Eigen3 3.3 REQUIRED)

include_directories(include)


add_executable(test_mmul test/mmul.cpp)
target_include_directories(test_mmul INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_mmul executor CONAN_PKG::doctest Eigen3::Eigen)

doctest_discover_tests(
        test_mmul
        EXTRA_ARGS
        -s)
