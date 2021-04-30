cmake_minimum_required(VERSION 3.0.0)
project(Ising_Cristi_omp VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(Ising_Cristi_omp main.cpp)

SET(GCC_OPENMP_FLAG    "-fopenmp")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_OPENMP_FLAG}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
