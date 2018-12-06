cmake_minimum_required(VERSION 3.9)
project(heat1D)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

find_package(MPI REQUIRED)

if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

include_directories(
    src
)

set(SOURCE_FILES
    src/main.cc
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(heat1D ${MPI_LIBRARIES})