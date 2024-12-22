cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(prim prim.cxx)
add_executable(dijkstra dijkstra.cxx)
add_executable(dijkstra_pq dijkstra_pq.cxx)
