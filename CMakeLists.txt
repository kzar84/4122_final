cmake_minimum_required(VERSION 3.9)
project(final_proj)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_subdirectory(part1)
add_subdirectory(part2)
add_subdirectory(part3)
add_subdirectory(sharedlib)