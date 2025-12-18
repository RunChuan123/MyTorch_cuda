add_library(ops)

target_sources(ops
PUBLIC
scalarops.cu
multiple.cu
)

target_include_directories(ops ${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(ops
PUBLIC 
tensor
)