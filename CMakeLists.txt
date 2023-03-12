cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.7")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(cuda_loop LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE "Release")
list(APPEND CMAKE_PREFIX_PATH "/home/xth/anaconda3/envs/pytorch3d/lib/python3.9/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

enable_language(CUDA)
AUX_SOURCE_DIRECTORY(src SOURCES)
#file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/*.cu")

add_library(${PROJECT_NAME} SHARED loop_cuda_torch.cpp ${SOURCES})
target_link_libraries( ${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} Python::Python)

