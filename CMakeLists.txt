# CMakeLists.txt for SOFT library, v2.0
#
# Adapted from Makefile.fftw1 to build libsoft1 only

cmake_minimum_required(VERSION 3.1)
project(soft VERSION 2.0 LANGUAGES C)

# Set default build type to Release (implies -O3)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Find FFTW3
find_package(FFTW3 CONFIG REQUIRED)
include_directories(${FFTW3_INCLUDE_DIRS})
find_library(FFTW3 fftw3)

# Find all source files
file(GLOB COMMON_SOURCES "common/*.c")
file(GLOB LIB_SOURCES "lib1/*.c")

# Specify where to find the headers
include_directories("include")

# Compile and link the library
add_library(soft1 ${COMMON_SOURCES} ${LIB_SOURCES})
target_link_libraries(soft1 ${FFTW3})

# Install the library into the lib directory
# (This is done relative to the CMAKE_INSTALL_PREFIX and is a sensible default)
install(TARGETS soft1 EXPORT soft-config DESTINATION lib)

# Install the headers into the include directory
install(FILES
    ${CMAKE_SOURCE_DIR}/include/makeweights.h
    ${CMAKE_SOURCE_DIR}/include/rotate_so3_fftw.h
    ${CMAKE_SOURCE_DIR}/include/rotate_so3_utils.h
    ${CMAKE_SOURCE_DIR}/include/s2_primitive.h
    ${CMAKE_SOURCE_DIR}/include/s2_cospmls.h
    ${CMAKE_SOURCE_DIR}/include/s2_legendreTransforms.h
    ${CMAKE_SOURCE_DIR}/include/s2_semi_fly.h
    ${CMAKE_SOURCE_DIR}/include/s2_semi_memo.h
    ${CMAKE_SOURCE_DIR}/include/soft_fftw.h
    ${CMAKE_SOURCE_DIR}/include/utils_so3.h
    ${CMAKE_SOURCE_DIR}/include/wrap_fftw.h
    DESTINATION include
)

# Export package information
export(
    TARGETS soft1
    FILE "${CMAKE_CURRENT_BINARY_DIR}/soft-config.cmake"
)
install(
    EXPORT soft-config
    DESTINATION "lib/cmake/soft"
)

