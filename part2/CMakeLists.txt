cmake_minimum_required(VERSION 3.9.1)
project(p32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
INCLUDE_DIRECTORIES(src)
LINK_DIRECTORIES(src)

set(SOURCE_FILES
    src/complex.cc
    src/complex.h
    src/input_image.cc
    src/input_image.h
    src/p32.cc
    src/fft.cc
    src/fft.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
