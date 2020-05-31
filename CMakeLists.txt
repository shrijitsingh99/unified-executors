cmake_minimum_required(VERSION 3.14)

project(unified_api_experiments VERSION 0.0.1 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)


include_directories(include)

add_executable(template_meta_plg src/template_meta_plg.cpp)
add_executable(executor src/executor.cpp)

target_link_libraries(executor Eigen3::Eigen OpenMP::OpenMP_CXX)
