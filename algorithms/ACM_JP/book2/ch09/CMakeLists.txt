cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(binary_search_tree binary_search_tree.cxx)
add_executable(stl_set stl_set.cxx)
add_executable(stl_map stl_map.cxx)
add_executable(dictionary dictionary.cxx)
