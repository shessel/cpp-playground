cmake_minimum_required(VERSION 3.5)
project(intrinsics)
add_executable(simd_vec simd_vec.cpp)
target_compile_options(simd_vec PRIVATE -Wall -Wextra -Werror -std=c++1z -march=native)
