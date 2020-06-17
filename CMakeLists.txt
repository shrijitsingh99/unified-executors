cmake_minimum_required(VERSION 3.8)

project(unified_executors VERSION 0.0.1)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(doctest REQUIRED)

enable_testing()

add_subdirectory(executor)
add_subdirectory(application)

add_executable(main_test test/main.cpp)
target_link_libraries(main_test doctest::doctest executor)

doctest_discover_tests(main_test)
