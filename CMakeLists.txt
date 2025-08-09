cmake_minimum_required(VERSION 3.20)

#Fuzzy search with Smith-Waterman algorithm
project(fuzzy_sw CXX)
set(CMAKE_CXX_COMPILER "clang++-20")
set(CMAKE_C_COMPILER "clang-20")

set(CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SRC main.cpp)

set(SRC main.cpp)
set(HDR )

add_executable(fuzzy_sw ${SRC} ${HDR})
target_compile_options(fuzzy_sw PUBLIC -stdlib=libc++ -msse4.1 -mavx2)
target_compile_definitions(fuzzy_sw PUBLIC _LIBCPP_ENABLE_EXPERIMENTAL)
target_link_options(fuzzy_sw PUBLIC  -stdlib=libc++ -L/usr/lib/llvm-20/lib)
target_link_libraries(fuzzy_sw PUBLIC c++abi)
set_property(TARGET fuzzy_sw PROPERTY CXX_STANDARD 23)

