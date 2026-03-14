cmake_minimum_required(VERSION 3.10)
project(MyYoloInference)

set(CMAKE_CXX_STANDARD 17)

add_library(MyYoloInference SHARED
    src/definitions.h
    src/global.h
    src/inferenceclassify.cpp
    src/inferenceclassify.h
    src/inference.cpp
    src/inferencedetect.cpp
    src/inferencedetect.h
    src/inferencefactory.cpp
    src/inferencefactory.h
    src/inference.h
    src/inferenceobb.cpp
    src/inferenceobb.h
    src/inferencepose.cpp
    src/inferencepose.h
    src/inferencesegment.cpp
    src/inferencesegment.h
    src/metadata.cpp
    src/metadata.h
    src/my-yolo-inference.cpp
    src/my-yolo-inference.h
    src/utils.cpp
    src/utils.h
)

target_include_directories(MyYoloInference
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

set_target_properties(MyYoloInference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME "MyYoloInference"
    PREFIX ""
)

add_subdirectory(vendor/base64)
find_package(OpenCV REQUIRED)
target_link_libraries(MyYoloInference
    PUBLIC
        base64
        ${OpenCV_LIBS}
)

add_custom_target(README SOURCES README.md)

option(BUILD_EXAMPLES "Build Examples" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_custom_command(
        TARGET MyYoloInference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/examples/res
            ${CMAKE_BINARY_DIR}/examples/res
    )
endif()
