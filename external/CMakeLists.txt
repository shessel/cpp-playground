cmake_minimum_required(VERSION 3.28.0 FATAL_ERROR)

project(cpp_playground)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
add_subdirectory(benchmark)