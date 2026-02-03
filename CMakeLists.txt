cmake_minimum_required(VERSION 3.10)
project(SpectralClustering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE
    ${PROJECT_SOURCE_DIR}/external/eigen-3.4.0
)

# Eigen
add_library(Spectra INTERFACE)
target_include_directories(Spectra INTERFACE
    ${PROJECT_SOURCE_DIR}/external/spectra-1.2.0/include
)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(spectral_clustering ${SRC_FILES})

target_compile_options(spectral_clustering PRIVATE -O3 -march=native)

target_link_libraries(spectral_clustering
    PRIVATE
        Eigen
        Spectra
        MPI::MPI_CXX
)
