#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install ROCm script does not support Windows or MacOS"
fi

script_msg "Setup Sanitizer"

export ASAN_OPTIONS="suppressions=${APCI_ALPAKA_ROOT}/sanitizer/asan_suppressions.txt"
export TSAN_OPTIONS="suppressions=${APCI_ALPAKA_ROOT}/sanitizer/tsan_suppressions.txt,ignore_noninstrumented_modules=1"
export LSAN_OPTIONS="suppressions=${APCI_ALPAKA_ROOT}/sanitizer/lsan_suppressions.txt"
export UBSAN_OPTIONS="suppressions=${APCI_ALPAKA_ROOT}/sanitizer/ubsan_suppressions.txt"
store_variable ASAN_OPTIONS
store_variable TSAN_OPTIONS
store_variable LSAN_OPTIONS
store_variable UBSAN_OPTIONS

if [[ "${APCI_SANITIZER_TSAN}" == "ON" ]]; then
    rnd_bits=$(sudo sysctl -n vm.mmap_rnd_bits)
    if [[ -z ${GITHUB_ACTIONS+x} ]] && [[ -z ${GITLAB_CI+x} ]]; then
        # local container
        if [[ "${rnd_bits}" != 28 ]]; then
            echo_yellow "[WARNING]: vm.mmap_rnd_bits is set to ${rnd_bits}. The TSAN may fail." \
                "Run 'sudo sysctl vm.mmap_rnd_bits=28' on the host to fix the issue."
        fi
    else
        if [[ "${rnd_bits}" == 28 ]]; then
            echo_green "vm.mmap_rnd_bits=28 is set"
        else
            echo_run sudo sysctl vm.mmap_rnd_bits=28
            if [[ $(sudo sysctl -n vm.mmap_rnd_bits) != 28 ]]; then
                exit_error "Cannot set vm.mmap_rnd_bits to 28"
            fi
        fi
    fi
fi
