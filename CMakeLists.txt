cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 17)
project("Sim")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#Profiling
#message(STATUS "${CMAKE_CURRENT_LIST_DIR}")
#message(STATUS "${GPERFTOOLS_PROFILER}")
#FIND_LIBRARY(PROFILER_LIB)

#set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
#LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
#include("GperftoolsConfig.cmake")

#link_directories(/usr/local/lib)
#set(Gperftools_DIR /home/juanraul/Desktop/software/gperftools)
#find_package(Gperftools REQUIRED)

#set(CMAKE_CXX_FLAGS "-DWITHGPERFTOOLS -g -lprofiler")
#include_directories(/usr/local/include/gperftools/)

include_directories(eigen3)
add_library(solver MoM_ob.cpp Solver.cpp)
target_link_libraries(solver)
add_executable(Sim genDist.cpp)
target_link_libraries(Sim solver libcomplex_bessel.so)

#Profiling
#MESSAGE(STATUS "Library path GPERFTOOLS_PROFILER is " ${GPERFTOOLS_PROFILER})
#target_link_libraries(Sim ${GPERFTOOLS_PROFILER})
