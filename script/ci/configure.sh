#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake configure (configure.sh)"

parse_compiler_version "$APCI_DEVICE_COMPILER"

CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-""}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

# TODO: remove me, if all install scripts are ported
if [[ "$compiler_name" == "gcc" || "$compiler_name" == "clang" || "$compiler_name" == "icpx" ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH
    load_variable_if_not_exist APCI_C_COMPILER
    load_variable_if_not_exist APCI_CXX_COMPILER

    CMAKE_ARGS=(
        -S "${APCI_ALPAKA_ROOT}"
        -B "/build"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -Dalpaka_COMPILE_PEDANTIC=ON
        -Dalpaka_DOCS=ON
        -Dalpaka_TESTS=ON
        -Dalpaka_BENCHMARKS=ON
        -Dalpaka_EXAMPLES=ON
        -DBUILD_TESTING=ON
        -Dalpaka_HEADERCHECKS=ON
        -Dalpaka_LOG=dynamic
        -Dalpaka_FAST_MATH=OFF
        "-DCMAKE_C_COMPILER=$APCI_C_COMPILER"
        "-DCMAKE_CXX_COMPILER=$APCI_CXX_COMPILER"
    )

    declare -A ap_deps=(
        ["alpaka_DEP_OMP"]=OFF
        ["alpaka_DEP_HWLOC"]=OFF
        ["alpaka_DEP_CUDA"]=OFF
        ["alpaka_DEP_HIP"]=OFF
        ["alpaka_DEP_ONEAPI"]=OFF
    )

    # if no GPU SDK is used
    if [[ ("$APCI_HIP" == 0) && ("$compiler_name" == "gcc" || "$compiler_name" == "clang") ]]; then
        ap_deps['alpaka_DEP_OMP']=ON
    fi

    if [[ "$APCI_HIP" != 0 ]]; then
        load_variable_if_not_exist ROCM_PATH

        export PATH=${ROCM_PATH}/bin:$PATH
        export PATH=${ROCM_PATH}/llvm/bin:$PATH
        export CMAKE_PREFIX_PATH=$ROCM_PATH:$CMAKE_PREFIX_PATH

        ap_deps['alpaka_DEP_HIP']=ON

        if [[ -n ${APCI_AMD_GPU_ARCH+x} ]]; then
            CMAKE_ARGS+=(
                -DCMAKE_HIP_ARCHITECTURES="${APCI_AMD_GPU_ARCH}"
                -DGPU_TARGETS="${APCI_AMD_GPU_ARCH}")
        fi
    fi

    if [[ "$APCI_ONEAPI" != 0 ]]; then
        load_variable_if_not_exist ONEAPI_PATH

        # shellcheck source=script/ci/install/oneapi/setvars.sh
        source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"

        ap_deps['alpaka_DEP_ONEAPI']=ON

        CMAKE_ARGS+=(-Dalpaka_EXEC_CpuSerial=OFF)

        declare -A ap_sycl_targets=(
            ["alpaka_ONEAPI_Cpu"]=OFF
            ["alpaka_ONEAPI_IntelGpu"]=OFF
            ["alpaka_ONEAPI_NvidiaGpu"]=OFF
            ["alpaka_ONEAPI_AmdGpu"]=OFF
        )

        if [[ "${APCI_ONEAPI_TARGET}" == "cpu" ]]; then
            ap_sycl_targets['alpaka_ONEAPI_Cpu']=ON
        elif [[ "${APCI_ONEAPI_TARGET}" == "intel_gpu" ]]; then
            ap_sycl_targets['alpaka_ONEAPI_IntelGpu']=ON
        else
            exit_error "APCI_ONEAPI_TARGET unknown value: ${APCI_ONEAPI_TARGET}.\n" \
                "Only supported values are: cpu, intel_gpu"
        fi

        if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
            export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2025.1/lib/:${APCI_ONEAPI}"
        fi

        for target in "${!ap_sycl_targets[@]}"; do
            CMAKE_ARGS+=("-D${target}=${ap_sycl_targets[$target]}")
        done
    fi

    for dep in "${!ap_deps[@]}"; do
        CMAKE_ARGS+=("-D${dep}=${ap_deps[$dep]}")
    done

    echo_green "$(echo_if_not_empty LD_LIBRARY_PATH)" \
        "${APCI_CMAKE_BIN_PATH}/cmake" \
        "${CMAKE_ARGS[*]}"
    if [[ -z ${GITHUB_ACTIONS+x} ]]; then
        "${APCI_CMAKE_BIN_PATH}/cmake" "${CMAKE_ARGS[@]}"
    fi
fi
