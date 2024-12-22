cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(floyd floyd.cxx)
add_executable(topological_sort_bfs topological_sort_bfs.cxx)
add_executable(topological_sort_dfs topological_sort_dfs.cxx)
add_executable(articulation_point articulation_point.cxx)
add_executable(tree_diameter tree_diameter.cxx)
add_executable(kruskal kruskal.cxx)
