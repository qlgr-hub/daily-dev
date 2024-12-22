cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(disjointset disjointset.cxx)
add_executable(kdtree kdtree.cxx)
