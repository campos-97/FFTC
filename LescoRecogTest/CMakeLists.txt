cmake_minimum_required(VERSION 3.9)
project(LescoRecogTest)

set(CMAKE_CXX_STANDARD 17)

add_executable(LescoRecogTest main.cpp MyVideoCapture.cpp MyVideoCapture.h)

FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(LescoRecogTest ${OpenCV_LIBS})