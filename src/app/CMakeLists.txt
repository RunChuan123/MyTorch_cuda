add_executable(minitorch_app
    main.cpp
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(minitorch_app PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(minitorch_app
    PRIVATE
        functional
        ops
        tensor
        utils
)
