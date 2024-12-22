cmake_minimum_required(VERSION  3.25)

project(main)

include_directories(../common/)
add_compile_options(-fPIC)

add_executable(prime prime.cxx)
add_executable(gcd gcd.cxx)
add_executable(mod_pow mod_pow.cxx)
