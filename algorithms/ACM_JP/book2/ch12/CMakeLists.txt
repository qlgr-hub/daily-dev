cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(graph graph.cxx)
add_executable(dfs_r dfs_r.cxx)
add_executable(dfs_s dfs_s.cxx)
add_executable(bfs bfs.cxx)
add_executable(connected_components connected_components.cxx)
