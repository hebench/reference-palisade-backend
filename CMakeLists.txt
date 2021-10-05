cmake_minimum_required(VERSION 3.12)
project(hebench_palisade_backend LANGUAGES C CXX)

# General Setup
if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
else()
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of Build" FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

###########################################################################################

option(HIDE_EXT_WARNINGS "Attempts to hide all warnings encountered by third-party projects" ON)

message(STATUS "HIDE_EXT_WARNINGS: ${HIDE_EXT_WARNINGS}")

###########################################################################################

# declare main backend target
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/third-party)

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "${FETCHCONTENT_BASE_DIR}/palisade/palisade-build/lib") # PALISADE strips rpath
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(${PROJECT_NAME} SHARED)

###########################################################################################

# third-party

# Must Set 4 Variable per third-party, have a _COMPONENT_NAME.cmake file ready, then include the import util.
# Goal is for cmake to fail out if particular library (lib + includes) dont exist/aren't found
## _COMPONENT_NAME: The name used to generate lib-specific variables, printing, and filename resolution
## _COMPONENT_HEADER: A path to a known header file relative to the imported include directory
## _COMPONENT_LIB_NAME: The name of the library that will be used to link to it
## _HEADER_ONLY: Whether or not the library being imported is header-only
## _EXTRA_[LIBS|INCLUDES]: Only neccessary if the 3rd-party lib being added requires it. Requires list of strings (see PALISADE below)


## API-BRIDGE
set(_COMPONENT_NAME "API_BRIDGE")
set(_COMPONENT_HEADER "hebench/api_bridge/api.h")
set(_COMPONENT_LIB_NAME "hebench_cpp")
set(_HEADER_ONLY FALSE)
include(cmake/utils/import-library.cmake)
include(cmake/third-party/${_COMPONENT_NAME}_check.cmake)

## PALISADE
set(_COMPONENT_NAME "PALISADE")
set(_COMPONENT_HEADER "palisade/pke/palisade.h")
set(_COMPONENT_LIB_NAME "PALISADEpke")
set(_HEADER_ONLY FALSE)
set(_EXTRA_LIBS "libPALISADEcore.so")
set(_EXTRA_INCLUDES "palisade/core;palisade/pke;palisade")
include(cmake/utils/import-library.cmake)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/include/engine/palisade_version.h.in1")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/include/engine/palisade_version.h.in1" "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_version.h")

###########################################################################################

# find packages required by our target (after third parties to avoid polluting them)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package (OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

###########################################################################################

# add our target source code

set(${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bfv/palisade_bfv_matmulteip_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bfv/palisade_bfv_matmultval_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bfv/palisade_bfv_matmult_cipherbatchaxis_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/bfv/palisade_bfv_matmultrow_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_eltwiseadd_pc_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_matmulteip_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_matmultval_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_matmult_cipherbatchaxis_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_matmultrow_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/benchmarks/ckks/palisade_ckks_logreg_benchmark.cpp"
    )

set(${PROJECT_NAME}_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bfv/palisade_bfv_matmulteip_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bfv/palisade_bfv_matmultval_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bfv/palisade_bfv_matmult_cipherbatchaxis_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/bfv/palisade_bfv_matmultrow_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_eltwiseadd_pc_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_matmulteip_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_matmultval_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_matmult_cipherbatchaxis_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_matmultrow_benchmark.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmarks/ckks/palisade_ckks_logreg_benchmark.h"
    )

# engine
list(APPEND ${PROJECT_NAME}_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_engine.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_context.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_error.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/engine/palisade_version.h"
    )

list(APPEND ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/palisade_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/palisade_context.cpp"
    )

target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

###########################################################################################

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
