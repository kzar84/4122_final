cmake_minimum_required(VERSION 3.9)
project(mylib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

add_library(mylib SHARED 
            complex.cc
            complex.h 
            input_image.cc 
            input_image.h
            ) 