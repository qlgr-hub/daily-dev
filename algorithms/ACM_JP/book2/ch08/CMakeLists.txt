cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(rooted_tree rooted_tree.cxx)
add_executable(binary_tree binary_tree.cxx)
add_executable(tree_walk tree_walk.cxx)
add_executable(reconstruction reconstruction.cxx)
