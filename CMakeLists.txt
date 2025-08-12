cmake_minimum_required(VERSION 3.20)

#Fuzzy search with Smith-Waterman algorithm
set(CXX_STANDARD 23)
project(fuzzy_sw CXX)
set(CMAKE_CXX_COMPILER "clang++-20")
set(CMAKE_C_COMPILER "clang-20")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SRC main.cpp fuzzy_sw_par.cpp fuzzy_sw_seq.cpp)
set(HDR fuzzy_sw.hpp)

add_executable(fuzzy_sw ${SRC} ${HDR})
# target_compile_options(fuzzy_sw PRIVATE -fsanitize=address)
# target_link_options(fuzzy_sw PRIVATE  -fsanitize=address)

target_compile_options(fuzzy_sw PUBLIC -stdlib=libc++ -msse4.1 -mavx2)
target_compile_definitions(fuzzy_sw PUBLIC _LIBCPP_ENABLE_EXPERIMENTAL)
target_link_options(fuzzy_sw PUBLIC  -stdlib=libc++ -L/usr/lib/llvm-20/lib)
target_link_libraries(fuzzy_sw PUBLIC c++abi)
set_property(TARGET fuzzy_sw PROPERTY CXX_STANDARD 23)

