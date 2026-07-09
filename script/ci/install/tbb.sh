#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

script_msg "Install Intel TBB"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install Intel TBB script does not support Windows or MacOS"
fi

: "${APCI_DEVICE_COMPILER?'The device compiler must be specified'}"

if [[ "${APCI_TBB}" == "ON" ]]; then
    parse_compiler_version "$APCI_DEVICE_COMPILER"

    if [[ "$compiler_name" != "icpx" ]]; then
        install_msg "Intel TBB $compiler_version"
        lazy_apt_update
        quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install -y libtbb-dev
    else
        echo_green "Use Intel TBB from OneAPI SDK"
    fi
else
    echo_green "Skipped install Intel TBB because it is not required for the job."
fi
