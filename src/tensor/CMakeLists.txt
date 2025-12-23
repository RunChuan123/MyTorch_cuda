add_library(tensor STATIC
    tensor.cu
)

set_target_properties(tensor PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src/tensor
)

target_link_libraries(tensor
    PUBLIC
        utils
)
