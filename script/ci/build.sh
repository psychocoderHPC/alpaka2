#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Run CMake build (build.sh)"

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}

if [[ "$APCI_ONEAPI" != 0 ]]; then
    if [[ ! "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
        export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2025.1/lib/:${LD_LIBRARY_PATH}"
    fi
fi

parse_compiler_version "$APCI_DEVICE_COMPILER"
# TODO: remove me, if all install scripts are ported
if [[ "$compiler_name" == "gcc" || "$compiler_name" == "clang" || "$compiler_name" == "icpx" ]]; then
    load_variable_if_not_exist APCI_CMAKE_BIN_PATH

    echo_green "$(echo_if_not_empty LD_LIBRARY_PATH)" \
        "${APCI_CMAKE_BIN_PATH}/cmake" \
        --build /build \
        "-j${APCI_BUILD_THREADS}"
    if [[ -z ${GITHUB_ACTIONS+x} ]]; then
        "${APCI_CMAKE_BIN_PATH}/cmake" --build /build "-j${APCI_BUILD_THREADS}"
    fi
fi
