cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(parallel_orthogonal geometry_object.cxx parallel_orthogonal.cxx)
add_executable(projection geometry_object.cxx projection.cxx)
add_executable(reflection geometry_object.cxx reflection.cxx)
