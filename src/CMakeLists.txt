add_library(MyYoloInference SHARED
    global.h
    my-yolo-inference.h
    my-yolo-inference.cpp
    vendor/base64.h
    vendor/base64.cpp
)

target_include_directories(MyYoloInference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MyYoloInference PRIVATE ${OpenCV_LIBS})

set_target_properties(MyYoloInference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME "MyYoloInference"
    PREFIX ""
)
