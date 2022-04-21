cmake_minimum_required(VERSION 3.0.0)
project(opencv_tt VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CTest)
enable_testing()

include_directories("C:/Users/TangTao/source/repos/github/opencv-4.x/build/install/include")

add_executable(opencv_tt main.cpp)

find_library(opencv_core NAMES opencv_core455d PATHS "C:/Users/TangTao/source/repos/github/opencv-4.x/build/install/x64/vc17/lib")
target_link_libraries(opencv_tt PUBLIC ${opencv_core})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
