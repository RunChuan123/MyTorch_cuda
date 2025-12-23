cmake_minimum_required(VERSION 3.20)

project(MiniTorch CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src/utils)
add_subdirectory(src/ops)
add_subdirectory(src/tensor)
add_subdirectory(src/functional)
add_subdirectory(src/app)



