cmake_minimum_required(VERSION 3.10)
project(MyYoloInference)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_subdirectory(src)
add_subdirectory(examples)

add_custom_target(README SOURCES README.md)
