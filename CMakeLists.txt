cmake_minimum_required(VERSION 3.20)
project(mytorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 让 VSCode 自动生成 compile_commands.json（方便 clangd）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")


# 找 CUDA Toolkit（auto）
find_package(CUDAToolkit REQUIRED)

# 所有代码文件（.cu 会被自动编译）
file(GLOB MYTORCH_CORE
    "${PROJECT_SOURCE_DIR}/*.cu"
    "${PROJECT_SOURCE_DIR}/*.cuh"
    "${PROJECT_SOURCE_DIR}/*.h"
    "${PROJECT_SOURCE_DIR}/*.cpp"
)

file(GLOB MYTORCH_OPS
    "${PROJECT_SOURCE_DIR}/ops/*.cuh"
    "${PROJECT_SOURCE_DIR}/ops/*.cu"
)

# 可执行文件
add_executable(main ${MYTORCH_CORE} ${MYTORCH_OPS})

# 包含 CUDA include 路径
target_include_directories(main PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

# 链接 CUDA runtime
target_link_libraries(main PRIVATE
    CUDA::cudart
)
