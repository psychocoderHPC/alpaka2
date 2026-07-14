#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install hwloc script does not support Windows or MacOS"
fi

script_msg "Install APCI_HWLOC"

if [[ "${APCI_HWLOC}" == "ON" ]]; then
    install_msg "hwloc"

    hwloc_package_list=(libhwloc-dev pkg-config)

    if [[ "$APCI_HIP" != 0 ]]; then
        case $(cat /etc/os-release) in
        # Ubuntu 22.04 is fine. No special handling.
        *"22.04"*) ;;
        *"24.04"*)
            # hwloc installs as dependency g++-13. The clang++ of the ROCm SDK detects g++-11 and
            # g++-13 and tries to use the libstdc++ of g++-13, which is not installed. Therefore
            # install the libstdc++-13 manually.
            hwloc_package_list+=(libstdc++-13-dev)
            ;;
        *) exit_error "hwloc + ROCm is not supported on this operations system container." ;;
        esac
    fi

    if dpkg -s "${hwloc_package_list[@]}" >/dev/null 2>&1; then
        echo_yellow "hwloc is already installed via apt. Skip installation."
    else
        lazy_apt_update
        quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install -y "${hwloc_package_list[@]}"
    fi
    echo_run which pkg-config
    echo "hwloc version: $(pkg-config --modversion hwloc)"
    echo_run dpkg -L libhwloc-dev
else
    echo_green "Skipped install hwloc because it is not required for the job."
fi
