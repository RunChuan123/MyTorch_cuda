add_library(functional
tensor_ops.cu
)

set_target_properties(functional PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(functional PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(functional
PUBLIC 
tensor
utils
ops
)