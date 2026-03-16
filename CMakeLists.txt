# CMAKE_CUDA_ARCHITECTURES <native> requires CMake 3.24.
cmake_minimum_required(VERSION 3.24)

project(
  CoScheduler
  VERSION 0.0.1
  LANGUAGES CXX)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_TRACCC "Build traccc" ON)
option(BUILD_PATATRACK "Build Patatrack" OFF)

# Require a C++20 compiler for C++.
set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED
    ON
    CACHE BOOL "Whether the C++ standard is required")
set(CMAKE_CXX_EXTENSIONS OFF)

# Require a C++17 support for CUDA.
set(CMAKE_CUDA_STANDARD
    20
    CACHE STRING "The (CUDA) C++ standard to use")
# Set the architecture to build code for.
set(CMAKE_CUDA_ARCHITECTURES
    native
    CACHE STRING "CUDA architectures to build device code for")

# Enable the CUDA language, now that relevant variables have been set.
enable_language(CUDA)
#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)
# Require a recent Boost that dependant traccc requires (for fiber).
# Tell the user what's happening.
message( STATUS "Building recent Boost as part of the scheduler tests" )

# Components required by the project.
set(BOOST_ENABLE_CMAKE ON )
set(BOOST_SKIP_INSTALL_RULES OFF)
set(BUILD_SHARED_LIBS ON)

FetchContent_Declare( Boost OVERRIDE_FIND_PACKAGE
   URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.tar.xz
   URL_HASH SHA256=aca59f889f0f32028ad88ba6764582b63c916ce5f77b31289ad19421a96c555f
   DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable( Boost )
find_package(Boost 1.90 REQUIRED)

# Stolen from Bosst's CMake files. Unclear why we should repeat here (needed by ACTS).
add_library(Boost::boost INTERFACE IMPORTED)
set_property(TARGET Boost::boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES Boost::headers)

# Set up the built-in externals.
add_subdirectory(externals)

find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(BUILD_PATATRACK)
  find_package(Patatrack REQUIRED)
endif()
if(BUILD_TRACCC)
  find_package(traccc REQUIRED)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS_DEBUG "-Wall -Wextra -g -O0 -fno-omit-frame-pointer")

# Add the libraries.

file(
  GLOB
  all_files
  CONFIGURE_DEPENDS
  "src/*.h"
  "src/*.cpp"
  "src/*.cuh"
  "src/*.cu")

add_library(CoScheduler SHARED ${all_files})
target_include_directories(
  CoScheduler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                     $<INSTALL_INTERFACE:include>)

target_link_libraries(CoScheduler PUBLIC TBB::tbb CUDA::cudart CUDA::cuda_driver Boost::headers)

if(BUILD_PATATRACK)
  # target_link_libraries(CoScheduler PUBLIC Patatrack::CUDACore
  # Patatrack::Backtrace)
endif()

add_subdirectory(components)

# Add the executables.

add_executable(schedule_simple bin/schedule_simple.cpp)
target_link_libraries(schedule_simple PRIVATE CoScheduler components_simple)

if(BUILD_TRACCC)
  # add_executable(schedule_traccc bin/schedule_traccc.cpp)
  # target_link_libraries(schedule_traccc PRIVATE CoScheduler components_traccc)
endif()

# Add the tests.
enable_testing()
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
