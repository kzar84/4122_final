cmake_minimum_required(VERSION 3.8)
project(p31)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

include_directories(${CMAKE_SOURCE_DIR}/shared) # include files from ProjectB

add_executable(${PROJECT_NAME} p31.cc)
target_link_libraries(${PROJECT_NAME} shared)