# src directory

#get the source files
file(GLOB_RECURSE c_source *.cpp *.cc *.h)
file(GLOB to_remove_main main.cpp)
list(REMOVE_ITEM c_source ${to_remove_main})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(c_lib STATIC ${c_source})