cmake_minimum_required(VERSION 3.14)

project(unified_api_experiments VERSION 0.0.1 LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(template_meta_plg src/template_meta_plg.cpp)
add_executable(sfinae_exec src/executor.cpp)
