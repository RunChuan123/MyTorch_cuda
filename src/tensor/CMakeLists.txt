add_library(tensor)

target_sources(tensor
PUBLIC
tensor.cu
)
target_include_directories(tensor ${CMAKE_CURRENT_SOURCE_DIR})

target_link_directories(tensor
PUBLIC 
utils
)