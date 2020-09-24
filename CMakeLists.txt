cmake_minimum_required(VERSION 3.16)
project(vmemtest LANGUAGES CXX CUDA)

##########################################################################
# Include Directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


find_package(CUDAToolkit REQUIRED)

##########################################################################
# Choose for which CC to build and if to enable Debug
option(CUDA_BUILD_CC35 "Build with compute capability 3.5 support" FALSE)
option(CUDA_BUILD_CC50 "Build with compute capability 5.0 support" FALSE)
option(CUDA_BUILD_CC52 "Build with compute capability 5.2 support" FALSE)
option(CUDA_BUILD_CC61 "Build with compute capability 6.1 support" FALSE)
option(CUDA_BUILD_CC70_SYNC "Build with compute capability 7.0 support - SYNC" FALSE)
option(CUDA_BUILD_CC70_ASYNC "Build with compute capability 7.0 support - ASYNC" TRUE)
option(CUDA_BUILD_CC75_SYNC "Build with compute capability 7.5 support - SYNC" FALSE)
option(CUDA_BUILD_CC75_ASYNC "Build with compute capability 7.5 support - ASYNC" FALSE)
option(CUDA_BUILD_INFO "Build with kernel statistics and line numbers" TRUE)
option(CUDA_BUILD_DEBUG "Build with kernel debug" FALSE)

##########################################################################
# CUDA Flags
if(CUDA_BUILD_CC35)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35")
endif ()
if (CUDA_BUILD_CC50)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
endif ()
if (CUDA_BUILD_CC52)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
endif ()
if (CUDA_BUILD_CC61)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_61,code=sm_61")
endif ()
if (CUDA_BUILD_CC70_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_70")
endif ()
if (CUDA_BUILD_CC70_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
endif ()
if (CUDA_BUILD_CC75_SYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_60,code=sm_75")
endif ()
if (CUDA_BUILD_CC75_ASYNC)
	string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75")
endif ()
if (CUDA_BUILD_INFO)
	string(APPEND CMAKE_CUDA_FLAGS " -keep --ptxas-options=-v -lineinfo")
endif ()
if (CUDA_BUILD_DEBUG)
	string(APPEND CMAKE_CUDA_FLAGS " -G")
endif ()

##########################################################################
# CXX Flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
string(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3")


##########################################################################
# Setup executable
file(GLOB CUDHDR *.cuh)
file(GLOB CUDSRC *.cu)
file(GLOB HDR *.h)
file(GLOB SRC *.cpp)
add_executable(vmemtest
    ${CUDHDR}
    ${CUDSRC}
	${HDR}
	${SRC}
)

set_property(TARGET vmemtest PROPERTY CXX_STANDARD 17)

##########################################################################
# Link libraries
target_link_libraries(vmemtest CUDA::cuda_driver)
