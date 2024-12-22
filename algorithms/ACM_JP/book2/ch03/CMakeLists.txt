cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(insertion_sort insertion_sort.cxx)
add_executable(bubble_sort bubble_sort.cxx)
add_executable(selection_sort selection_sort.cxx)
add_executable(stable_sort stable_sort.cxx)
add_executable(shell_sort shell_sort.cxx)
