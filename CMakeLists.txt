#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Axel Hübl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci,
#                Bernhard Manfred Gruber, Aurora Perego
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.20)
project(alpaka CXX)

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# alpaka_COMPILER_OPTIONS_HOST, alpaka_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'.")
    endif()
    if(type STREQUAL "list")
        set(${name} ${${name}} ${ARGN} PARENT_SCOPE)
    elseif(type STREQUAL "var")
        foreach(arg IN LISTS ARGN)
            set(tmp "${tmp} ${arg}")
        endforeach()
        set(${name} "${${name}} ${tmp}" PARENT_SCOPE)
    elseif(type STREQUAL "target")
        foreach(arg IN LISTS ARGN)
            target_compile_options(${name} INTERFACE ${arg})
        endforeach()
    else()
        message(FATAL_ERROR "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'.")
    endif()
endfunction()

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED alpaka_${name})
        set(alpaka_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE alpaka_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

set(CMAKE_CXX_STANDARD 20)

# This file's directory.
set(_alpaka_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})
include("${_alpaka_ROOT_DIR}/cmake/addExecutable.cmake")

# Add alpaka_ADD_LIBRARY function.
include("${_alpaka_ROOT_DIR}/cmake/addLibrary.cmake")

include(CheckLanguage) # check for CUDA/HIP language support

add_library(alpaka INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

option(alpaka_API_CUDA "Enable the CUDA GPU back-end" OFF)
option(alpaka_API_OMP "Enable the OMP back-end" ON)

## OpenMP
if(alpaka_API_OMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(alpaka INTERFACE OpenMP::OpenMP_CXX)
endif()

## CUDA
if(alpaka_API_CUDA)
    check_language(CUDA)
    alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

    option(alpaka_CUDA_EXPT_EXTENDED_LAMBDA "Enable CUDA extended lambda support " ON)
    if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>")
    endif()

    alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)

    if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
    endif()

    alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)
    if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--keep>")
    endif()

    option(alpaka_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx." OFF)
    if(alpaka_CUDA_SHOW_CODELINES)
        alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--source-in-ptx -lineinfo>")

        # This is shaky - We currently don't have a way of checking for the host compiler ID.
        # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -rdynamic>")
        endif()
        set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif()

    if(alpaka_API_OMP)
        if(NOT MSVC)
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>")

            # See https://github.com/alpaka-group/alpaka/issues/1755
            if((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND
            (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
                message(STATUS "clang >= 13 detected. Force-setting OpenMP to version 4.5.")
                alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp-version=45>")
            endif()
        else()
            alpaka_set_compiler_options(DEVICE target alpaka "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /openmp>")
        endif()
    endif()
endif()

#-------------------------------------------------------------------------------
# Include mdspan

set(alpaka_USE_MDSPAN "FETCH" CACHE STRING "Use std::mdspan with alpaka")
set_property(CACHE alpaka_USE_MDSPAN PROPERTY STRINGS "SYSTEM;FETCH;OFF")

if (alpaka_USE_MDSPAN STREQUAL "SYSTEM")
    find_package(mdspan REQUIRED)
    target_link_libraries(alpaka INTERFACE std::mdspan)
    target_compile_definitions(alpaka INTERFACE ALPAKA_USE_MDSPAN)
elseif (alpaka_USE_MDSPAN STREQUAL "FETCH")
    include(FetchContent)
    FetchContent_Declare(
            mdspan
            GIT_REPOSITORY https://github.com/kokkos/mdspan.git
            GIT_TAG 973ef6415a6396e5f0a55cb4c99afd1d1d541681
    )
    # we don't use FetchContent_MakeAvailable(mdspan) since it would also install mdspan
    # see also: https://stackoverflow.com/questions/65527126/how-to-disable-installation-a-fetchcontent-dependency
    FetchContent_GetProperties(mdspan)
    if(NOT mdspan_POPULATED)
        FetchContent_Populate(mdspan)
        if(${CMAKE_VERSION} VERSION_LESS "3.25.0")
            add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL)
        else()
            add_subdirectory(${mdspan_SOURCE_DIR} ${mdspan_BINARY_DIR} EXCLUDE_FROM_ALL SYSTEM)
        endif()
    endif()
    if(${CMAKE_VERSION} VERSION_LESS "3.25.0")
        get_target_property(mdspan_include_dir std::mdspan INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(alpaka SYSTEM INTERFACE ${mdspan_include_dir})
    else()
        target_link_libraries(alpaka INTERFACE std::mdspan)
    endif()
    target_compile_definitions(alpaka INTERFACE ALPAKA_USE_MDSPAN)
elseif (alpaka_USE_MDSPAN STREQUAL "OFF")
else()
    message(FATAL_ERROR "Invalid option for alpaka_USE_MDSPAN")
endif()

if (NOT alpaka_USE_MDSPAN STREQUAL "OFF")
    if (MSVC AND (alpaka_CXX_STANDARD LESS 20))
        message(WARNING "std::mdspan on MSVC requires C++20. Please enable C++20 via alpaka_CXX_STANDARD. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif ()

    if (alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
        # this issue actually only occurs when the host compiler (not the CXX compiler) is clang, but cmake does not let us query the host compiler id
        # see: https://gitlab.kitware.com/cmake/cmake/-/issues/20901
        message(WARNING "std::mdspan does not work with nvcc and clang as host compiler. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif ()

    if (alpaka_API_CUDA AND (NOT alpaka_CUDA_EXPT_EXTENDED_LAMBDA STREQUAL ON))
        message(WARNING "std::mdspan requires nvcc's extended lambdas. Use of std::mdspan has been disabled.")
        set(alpaka_USE_MDSPAN "OFF" CACHE STRING "Use std::mdspan with alpaka" FORCE)
    endif()
endif()



add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# alpaka IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE alpakaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("alpakaIde" SOURCES ${alpakaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/alpaka" FILES ${alpakaSources})

option(alpaka_TESTING "Enable/Disable testing" OFF)
option(alpaka_BENCHMARK "Enable/Disable benchmarks" OFF)
option(alpaka_EXAMPLES "Enable/Disable benchmarks" OFF)
include(CTest)

if (BUILD_TESTING OR alpaka_BENCHMARK)
    option(alpaka_SYSTEM_CATCH2 "Use the system provided Catch2." ON)
    if (alpaka_SYSTEM_CATCH2)
        find_package(Catch2 3.5.3 REQUIRED)
        include(Catch)
    else()
        # get Catch2 v3 and build it from source with the same C++ standard as the tests
        Include(FetchContent)
        FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.5.3)
        FetchContent_MakeAvailable(Catch2)
        target_compile_features(Catch2 PUBLIC cxx_std_20)
        include(Catch)

        # hide Catch2 cmake variables by default in cmake gui
        get_cmake_property(variables VARIABLES)
        foreach (var ${variables})
            if (var MATCHES "^CATCH_")
                mark_as_advanced(${var})
            endif()
        endforeach()
    endif()

    if (BUILD_TESTING)
        file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/tests/**")
        alpaka_add_executable(tests ${testSources})
        catch_discover_tests(tests)
        source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/tests" FILES ${testSources})
        target_compile_features(tests PRIVATE cxx_std_20)
        target_link_libraries(tests PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
    endif()

    if (alpaka_BENCHMARK)
        file(GLOB_RECURSE benchSources "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/**")
        alpaka_add_executable(bench ${benchSources})
        if (BUILD_TESTING)
            catch_discover_tests(bench)
        endif()
        target_link_libraries(bench PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
        target_compile_features(bench PRIVATE cxx_std_20)
        target_compile_definitions(bench PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)
    endif()
endif()

if(alpaka_EXAMPLES)
    add_subdirectory(example)
endif()

option(alpaka_EXEC_CpuSerial "Enable/Disable serial exeexecutor" ON)
if(NOT alpaka_EXEC_CpuSerial)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuSerial)
endif()
option(alpaka_EXEC_CpuOmpBlocks "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_CpuOmpBlocks)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
endif()
option(alpaka_EXEC_CpuOmpBlocksAndThreads "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_CpuOmpBlocksAndThreads)
    target_compile_definitions(alpaka INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocksAndThreads)
endif()
option(alpaka_EXEC_CpuCuda "Enable/Disable openMP blocks exeexecutor" ON)
if(NOT alpaka_EXEC_CpuCuda)
    target_compile_definitions(alpaka INTERFACE alpaka_DISABLE_EXEC_CpuCuda)
endif()
