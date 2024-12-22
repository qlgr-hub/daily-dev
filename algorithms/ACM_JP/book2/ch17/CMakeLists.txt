cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(coin_changing coin_changing.cxx)
add_executable(knapsack knapsack.cxx)
add_executable(LIS LIS.cxx)
add_executable(largest_square largest_square.cxx)
add_executable(largest_rectangle largest_rectangle.cxx)
