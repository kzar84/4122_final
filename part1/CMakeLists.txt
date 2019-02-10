cmake_minimum_required(VERSION 3.9.1)
project(p31)
find_package(Threads)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
INCLUDE_DIRECTORIES(src)
LINK_DIRECTORIES(src)
set(SOURCE_FILES
    src/complex.cc
    src/complex.h
    src/input_image.cc
    src/input_image.h
    src/p31.cc
)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries (${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
