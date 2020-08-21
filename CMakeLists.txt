cmake_minimum_required(VERSION 3.5)

project(unified_executors VERSION 0.3)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_subdirectory(executor)
add_subdirectory(application)

add_executable(main_test test/main.cpp)
target_link_libraries(main_test CONAN_PKG::doctest executor)

doctest_discover_tests(main_test)
