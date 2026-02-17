cmake_minimum_required(VERSION 3.20)
project(cublas_wrapper C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find CUDA install directory
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    if(DEFINED ENV{CUDA_PATH})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    elseif(EXISTS "/usr/local/cuda")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    elseif(EXISTS "/usr/local/cuda-13.1")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-13.1")
    elseif(EXISTS "/usr/local/cuda-12")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12")
    elseif(EXISTS "/usr/local/cuda-11")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11")
    else()
        message(FATAL_ERROR "Could not find CUDA installation. Set CUDA_TOOLKIT_ROOT_DIR explicitly.")
    endif()
endif()

message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

# Set CUDA paths
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIB_DIR: ${CUDA_LIB_DIR}")

# Source files
set(SOURCES cublas_wrapper.c)

# Create shared library
add_library(cublas_wrapper SHARED ${SOURCES})

# Include CUDA headers
target_include_directories(cublas_wrapper
    PRIVATE
        ${CUDA_INCLUDE_DIRS}
)

# Link CUDA libraries
target_link_directories(cublas_wrapper PRIVATE ${CUDA_LIB_DIR})
target_link_libraries(cublas_wrapper
    PRIVATE
        cudart
        cublas
)

# Set compiler flags
target_compile_options(cublas_wrapper PRIVATE -fPIC -O2)

# Set library output directory
set_target_properties(cublas_wrapper PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SOVERSION 1
)

# Installation rules
install(TARGETS cublas_wrapper
    LIBRARY DESTINATION lib
)

install(FILES cublas_wrapper.h
    DESTINATION include
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Output library: ${CMAKE_BINARY_DIR}/lib/libcublas_wrapper.so")
