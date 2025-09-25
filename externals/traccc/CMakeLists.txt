# CMake include(s).
cmake_minimum_required( VERSION 3.24 )
include( FetchContent )

# Tell the user what's happening.
message( STATUS "Building traccc as part of the project" )

# Optionally use a local traccc clone if provided
set(TRACCC_SOURCE_DIR "" CACHE PATH "Use a local traccc source directory instead of downloading")

if(TRACCC_SOURCE_DIR)
    message(STATUS "Using local traccc source at ${TRACCC_SOURCE_DIR} via FetchContent SOURCE_DIR")
    set(FETCHCONTENT_SOURCE_DIR_TRACCC "${TRACCC_SOURCE_DIR}")
endif()

# Optionally use a local detray clone if provided
set(DETRAY_SOURCE_DIR "" CACHE PATH "Use a local detray source directory instead of downloading")

if(DETRAY_SOURCE_DIR)
    message(STATUS "Using local detray source at ${DETRAY_SOURCE_DIR} via FetchContent SOURCE_DIR")
    set(FETCHCONTENT_SOURCE_DIR_DETRAY "${DETRAY_SOURCE_DIR}")
endif()

# Declare where to get traccc from.
set( SCHED_TRACCC_SOURCE
   "URL;https://github.com/acts-project/traccc/archive/refs/tags/v0.17.0.tar.gz;URL_MD5;fdfa678dc60005555edd0b9575873fa3"
   #"GIT_REPOSITORY;https://github.com/acts-project/traccc.git;GIT_TAG;v0.17.0"
   CACHE STRING "Source for traccc, when built as part of this project" )
mark_as_advanced( SCHED_TRACCC_SOURCE )
FetchContent_Declare( traccc ${SCHED_TRACCC_SOURCE} OVERRIDE_FIND_PACKAGE )

# Options used in the build of traccc.
set( TRACCC_BUILD_TESTING TRUE CACHE BOOL
   "Turn off the build of the traccc unit tests" )
set( TRACCC_BUILD_BENCHMARKS TRUE CACHE BOOL
   "Turn off the build of the traccc benchmarks" )
set( TRACCC_BUILD_EXAMPLES TRUE CACHE BOOL
   "Turn off the build of the traccc examples" )
set( TRACCC_USE_ROOT FALSE CACHE BOOL
   "Don't use ROOT as part of the traccc build" )
set( TRACCC_SETUP_TBB FALSE CACHE BOOL
   "Don't set up TBB as part of the traccc build" )
set( TRACCC_BUILD_CUDA TRUE CACHE BOOL
   "Turn on the build of traccc::cuda" )

# Get it into the current directory.
FetchContent_MakeAvailable( traccc )
