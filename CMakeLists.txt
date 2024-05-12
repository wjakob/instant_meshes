cmake_minimum_required(VERSION 3.10.0)

project(InstantMeshes VERSION 1.0.0 DESCRIPTION "Instant Meshes Library")

find_package(Eigen3 3.4 REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/install/eigen NO_DEFAULT_PATH)

# Build TBB
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb ext_build/tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")

option(BUILD_SHARED_LIBS "Build Open3D as a shared library." OFF)

set(LIBRARY_NAME InstantMeshes)

if(BUILD_SHARED_LIBS)
    set(LIBRARY_TYPE SHARED)
else()
    set(LIBRARY_TYPE STATIC)
endif()

# Add the library.
add_library(${LIBRARY_NAME} ${LIBRARY_TYPE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/adjacency.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bvh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cleanup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dedge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/extract.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hierarchy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshio.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshstats.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/normal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smoothcurve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subdivide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply/rply.c)

target_include_directories(${LIBRARY_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/dset
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pcg32
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pss
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/tbb/include)

target_link_libraries(${LIBRARY_NAME} PRIVATE
    Eigen3::Eigen
    tbb_static
    pthread
    dl)

# Set the library version
set_target_properties(${LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

if(BUILD_SHARED_LIBS)
    # Set .so major version so that libOpen3D.so.0 will be a symlink to libOpen3D.so.0.15.2
    set_target_properties(${LIBRARY_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# Install and export

# Set install directories
include(GNUInstallDirs)

set(LIBRARY_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/${LIBRARY_NAME}/cmake)
set(LIBRARY_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIBRARY_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})
set(LIBRARY_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
#set(LIBRARY_DOC_DIR ${CMAKE_INSTALL_DOCDIR})

# Create CMake config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in
    ${LIBRARY_NAME}Config.cmake
    INSTALL_DESTINATION ${LIBRARY_CMAKE_DIR}
    PATH_VARS LIBRARY_INCLUDE_DIR LIBRARY_LIBRARY_DIR
    NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
    ${LIBRARY_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

# Set install include directories
target_include_directories(${LIBRARY_NAME} INTERFACE
    $<INSTALL_INTERFACE:${LIBRARY_INCLUDE_DIR}>)


# Install library binaries
install(TARGETS ${LIBRARY_NAME} tbb_static
    EXPORT ${LIBRARY_NAME}Targets 
    LIBRARY DESTINATION ${LIBRARY_LIBRARY_DIR}
    RUNTIME DESTINATION ${LIBRARY_RUNTIME_DIR})

# Install CMake config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}ConfigVersion.cmake
    DESTINATION ${LIBRARY_CMAKE_DIR})

install(EXPORT ${LIBRARY_NAME}Targets
   NAMESPACE ${LIBRARY_NAME}::
   DESTINATION ${LIBRARY_CMAKE_DIR})

# Install public API

# Install the top-level header files
install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/aabb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/adjacency.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bvh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cleanup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dedge.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/extract.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/field.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hierarchy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshio.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshstats.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/normal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reorder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/smoothcurve.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/subdivide.h
    DESTINATION ${LIBRARY_INCLUDE_DIR}/instant-meshes)

# Install licence file
#install(FILES LICENSE DESTINATION ${LIBRARY_DOC_DIR})

# Uninstall
if(NOT TARGET uninstall)
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/uninstall.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake)
endif()
