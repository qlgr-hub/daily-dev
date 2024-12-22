cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(stack stack.cxx)
add_executable(queue queue.cxx)
add_executable(list list.cxx)
add_executable(square square.cxx)
