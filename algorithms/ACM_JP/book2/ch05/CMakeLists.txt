cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(linear_search linear_search.cxx)
add_executable(binary_search binary_search.cxx)
add_executable(hash_search hash_search.cxx)
add_executable(iterator iterator.cxx)
add_executable(lower_bound lower_bound.cxx)
add_executable(allocation allocation.cxx)
