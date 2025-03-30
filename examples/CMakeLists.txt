add_executable(test_implicit test_implicit.cpp)
target_include_directories(test_implicit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_implicit PRIVATE MyYoloInference ${OpenCV_LIBS})

add_executable(test_explicit test_explicit.cpp)
target_include_directories(test_explicit PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_explicit PRIVATE ${OpenCV_LIBS})

set_target_properties(test_implicit test_explicit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
