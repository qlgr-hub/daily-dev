cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)

add_executable(complete_binary_tree complete_binary_tree.cxx)
add_executable(maximum_heap maximum_heap.cxx)
add_executable(priority_queue priority_queue.cxx)
add_executable(stl_priority_queue stl_priority_queue.cxx)
add_executable(stl_priority_queue_2 stl_priority_queue_2.cxx)
