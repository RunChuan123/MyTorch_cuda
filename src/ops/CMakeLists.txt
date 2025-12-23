add_library(ops STATIC
    scalarops.cu
    multiple.cu
)

set_target_properties(ops PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(ops
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ops
    PUBLIC
        tensor
        utils
)
