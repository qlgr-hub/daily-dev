cmake_minimum_required(VERSION 3.20)

# 包含工具链文件
set(CMAKE_TOOLCHAIN_FILE "/workspace/toolchains/riscv.toolchain.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(demo_asm ASM)
add_executable(demo_asm.elf main.s)
