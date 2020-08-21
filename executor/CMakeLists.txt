cmake_minimum_required(VERSION 3.5)

project(executor VERSION 0.3)

find_package(OpenMP QUIET)

if(OpenMP_CXX_FOUND)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
endif()

add_library(${PROJECT_NAME} STATIC src/property/blocking.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} CONAN_PKG::doctest ${OpenMP_LIBS})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
