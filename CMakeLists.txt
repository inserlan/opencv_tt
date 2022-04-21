cmake_minimum_required(VERSION 3.0.0)
project(opencv_tt VERSION 0.1.0)

include(CTest)
enable_testing()

include_directories("D:/library/opencv-4.x/build/install/include")

add_executable(opencv_tt main.cpp)

find_library(opencv_core NAMES opencv_core455d PATHS "D:/library/opencv-4.x/build/install/x64/vc17/lib")
target_link_libraries(opencv_tt PUBLIC ${opencv_core})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
