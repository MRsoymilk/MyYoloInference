option(BUILD_TEST_IMPLICIT "Build test_implicit" ON)
option(BUILD_TEST_EXPLICIT "Build test_explicit" ON)
option(BUILD_TEST_BINARY_INPUT "Build test_binary_input" ON)
option(BUILD_TEST_VIDEO "Build test_video" ON)

if(BUILD_TEST_IMPLICIT)
  add_executable(test_implicit test_implicit.cpp)
  target_include_directories(test_implicit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(test_implicit PRIVATE MyYoloInference ${OpenCV_LIBS})
  list(APPEND TEST_TARGETS test_implicit)
endif()

if(BUILD_TEST_EXPLICIT)
  add_executable(test_explicit test_explicit.cpp)
  target_include_directories(test_explicit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(test_explicit PRIVATE ${OpenCV_LIBS})
  list(APPEND TEST_TARGETS test_explicit)
endif()

if(BUILD_TEST_BINARY_INPUT)
  add_executable(test_binary_input test_binary_input.cpp)
  target_include_directories(test_binary_input PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(test_binary_input PRIVATE MyYoloInference ${OpenCV_LIBS})
  list(APPEND TEST_TARGETS test_binary_input)
endif()

if(BUILD_TEST_VIDEO)
  add_executable(test_video test_video.cpp)
  target_include_directories(test_video PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(test_video PRIVATE MyYoloInference ${OpenCV_LIBS})
  list(APPEND TEST_TARGETS test_video)
endif()

if(TEST_TARGETS)
  set_target_properties(${TEST_TARGETS} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()

