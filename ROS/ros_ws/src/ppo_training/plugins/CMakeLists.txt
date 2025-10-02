cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dyn_obst_plugin)

find_package(gazebo REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(dyn_obst_plugin SHARED dyn_obst_plugin.cpp)
target_link_libraries(dyn_obst_plugin ${GAZEBO_LIBRARIES})
