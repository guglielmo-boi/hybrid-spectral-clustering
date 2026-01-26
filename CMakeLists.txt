cmake_minimum_required(VERSION 3.10)
project(SpectralClustering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen
add_library(Eigen5 INTERFACE)
target_include_directories(Eigen5 INTERFACE
    ${PROJECT_SOURCE_DIR}/external/eigen-5.0.1
)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# OpenMP
find_package(OpenMP REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(SOURCES
    src/csv.cpp
    src/k_means.cpp
    src/main.cpp
    src/spectral_clustering.cpp
)

add_executable(spectral_clustering ${SOURCES})

target_compile_options(spectral_clustering PRIVATE -O3 -march=native)

target_link_libraries(spectral_clustering
    PRIVATE
        Eigen5
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
)
