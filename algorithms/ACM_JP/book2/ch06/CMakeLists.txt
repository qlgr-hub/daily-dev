cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(recursive recursive.cxx)
add_executable(exhaustive_search exhaustive_search.cxx)
add_executable(koch koch.cxx)
