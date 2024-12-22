cmake_minimum_required(VERSION 3.6)

# C compiler
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_C_FLAGS "-Wall -nostdinc++ -nostdlib++")  # use libc++
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O4 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_C_STANDARD_LIBRARIES "-lc++")
set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES "/usr/include/c++/v1/")


# C++ compiler
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_CXX_FLAGS "-Wall -nostdinc++ -nostdlib++")  # use libc++
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
set(CMAKE_CXX_STANDARD_LIBRARIES "-lc++")
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES "/usr/include/c++/v1/")


# other tools
set(CMAKE_AR "/usr/bin/llvm-ar-18")
set(CMAKE_LINKER "/usr/bin/ld.lld-18")
set(CMAKE_NM "/usr/bin/llvm-nm-18")
set(CMAKE_OBJDUMP "/usr/bin/llvm-objdump-18")
set(CMAKE_RANLIB "/usr/bin/llvm-ranlib-18")
