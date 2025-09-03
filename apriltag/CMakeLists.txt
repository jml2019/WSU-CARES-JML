cmake_minimum_required(VERSION 3.10)
project(apriltag_tracking)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/local/include/apriltag
)

link_directories(
    /usr/local/lib
)

add_executable(apriltag_tracking apriltag_tracking.cpp)

target_link_libraries(apriltag_tracking
    ${OpenCV_LIBS}
    apriltag
    m
)

