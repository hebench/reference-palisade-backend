set(CONTENT_NAME "palisade")

set(CMAKE_CXX_FLAGS_CURRENT "${CMAKE_CXX_FLAGS}")

FetchContent_Declare(
  ${CONTENT_NAME}
  GIT_REPOSITORY https://gitlab.com/palisade/palisade-release.git
  GIT_TAG        ${${_COMPONENT_NAME}_TAG}
  SUBBUILD_DIR   ${FETCHCONTENT_BASE_DIR}/${CONTENT_NAME}/${CONTENT_NAME}-subbuild
  SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/${CONTENT_NAME}/${CONTENT_NAME}-src
  BINARY_DIR     ${FETCHCONTENT_BASE_DIR}/${CONTENT_NAME}/${CONTENT_NAME}-build
)

FetchContent_GetProperties(${CONTENT_NAME})
if(NOT ${CONTENT_NAME}_POPULATED)
  FetchContent_Populate(${CONTENT_NAME})
  set(WITH_OPENMP OFF CACHE BOOL "Disable using OpenMP")
  if (${HIDE_EXT_WARNINGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  endif()
  
  execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/third-party/prep-palisade-cmake.sh ${${CONTENT_NAME}_SOURCE_DIR}/CMakeLists.txt WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/third-party)
  add_subdirectory(${${CONTENT_NAME}_SOURCE_DIR} ${${CONTENT_NAME}_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_CURRENT}")
endif()

