cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7")
project(cuda_loop LANGUAGES CXX CUDA)
enable_language(CUDA)

#file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

add_executable(${PROJECT_NAME} test.cu)
