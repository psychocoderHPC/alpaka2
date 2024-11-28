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

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# alpaka IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE alpakaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("alpakaIde" SOURCES ${alpakaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/alpaka" FILES ${alpakaSources})

option(alpaka_TESTING "Enable/Disable testing" OFF)
option(alpaka_BENCHMARKS "Enable/Disable benchmarks" OFF)
option(alpaka_EXAMPLES "Enable/Disable examples" OFF)
include(CTest)

if (alpaka_TESTING OR alpaka_BENCHMARKS)
    option(alpaka_SYSTEM_CATCH2 "Use the system provided Catch2." OFF)
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

    if (alpaka_TESTING)
        file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/tests/**")
        alpaka_add_executable(tests ${testSources})
        catch_discover_tests(tests)
        source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/tests" FILES ${testSources})
        target_compile_features(tests PRIVATE cxx_std_20)
        target_link_libraries(tests PRIVATE Catch2::Catch2WithMain alpaka::alpaka)
    endif()

    if (alpaka_BENCHMARKS)
        file(GLOB_RECURSE benchSources "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/**")
        alpaka_add_executable(bench ${benchSources})
        if (alpaka_TESTING)
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
