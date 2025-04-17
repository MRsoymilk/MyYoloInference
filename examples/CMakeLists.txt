add_executable(test_implicit test_implicit.cpp)
target_include_directories(test_implicit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_implicit PRIVATE MyYoloInference ${OpenCV_LIBS})

add_executable(test_explicit test_explicit.cpp)
target_include_directories(test_explicit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_explicit PRIVATE ${OpenCV_LIBS})

add_executable(test_binary_input test_binary_input.cpp)
target_include_directories(test_binary_input PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_binary_input PRIVATE MyYoloInference ${OpenCV_LIBS})

set_target_properties(test_implicit test_explicit test_binary_input PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
