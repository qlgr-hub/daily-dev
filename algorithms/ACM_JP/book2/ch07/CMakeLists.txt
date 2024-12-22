cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(merge_sort merge_sort.cxx)
add_executable(partiton partiton.cxx)
add_executable(quick_sort quick_sort.cxx)
add_executable(counting_sort counting_sort.cxx)
add_executable(stl_sort stl_sort.cxx)
add_executable(inversions inversions.cxx)
add_executable(mimimum_cost mimimum_cost.cxx)
