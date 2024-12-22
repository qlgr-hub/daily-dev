cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(exhaustive_search exhaustive_search.cxx)
add_executable(fibonacci fibonacci.cxx)
add_executable(LCS LCS.cxx)
add_executable(matrix_chain_multiplication matrix_chain_multiplication.cxx)
