cmake_minimum_required(VERSION 3.9)
project(p32)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CMAKE_SOURCE_DIR}/mylib) # include files from ProjectB

add_executable(${PROJECT_NAME} p32.cc)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES} mylib)