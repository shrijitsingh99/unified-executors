cmake_minimum_required(VERSION 3.8)

project(executor VERSION 0.0.1 LANGUAGES CXX)

find_package(OpenMP QUIET)
find_package(CUDA QUIET)

if(OpenMP_CXX_FOUND)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
endif()

if (NOT CUDA_FOUND)
    set(CUDA_LIBRARIES "")
    set(CUDA_INCLUDE_DIRS "")
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE ${OpenMP_LIBS} ${CUDA_LIBRARIES})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
