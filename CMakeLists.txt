cmake_minimum_required(VERSION 3.8)

project(unified_executors VERSION 0.0.1)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
add_subdirectory(executor)
add_subdirectory(application)