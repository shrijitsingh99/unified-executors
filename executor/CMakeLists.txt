cmake_minimum_required(VERSION 3.8)

project(executor VERSION 0.3)

find_package(OpenMP QUIET)
find_package(CUDA QUIET)

if(OpenMP_CXX_FOUND)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
endif()

if (NOT CUDA_FOUND)
    set(CUDA_LIBRARIES "")
    set(CUDA_INCLUDE_DIRS "")
else()
    enable_language(CUDA)
endif()

add_library(${PROJECT_NAME} STATIC src/property/blocking.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} CONAN_PKG::doctest ${OpenMP_LIBS} ${CUDA_LIBRARIES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

if (CUDA_FOUND)
    # Lambda support in CUDA requires the --expt-extended-lambda flag to be explicitly specified
    target_compile_options(${PROJECT_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler -fopenmp>)
endif ()
