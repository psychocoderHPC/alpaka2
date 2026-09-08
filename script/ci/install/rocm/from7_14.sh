#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

echo_green "ROCm install method starting with version 7.14"

sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - |
    gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg >/dev/null

# Prevents apt warnings when the script is run a second time.
# Delete and recreate the source list to ensure that the correct apt sources are set.
if [[ -f /etc/apt/sources.list.d/rocm.list ]]; then
    sudo rm -rf /etc/apt/sources.list.d/rocm.list
fi

# require to set environment variable VERSION_ID
source /etc/os-release

sudo tee /etc/apt/sources.list.d/rocm.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu${VERSION_ID//./} stable main
EOF

retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

# If configured, install rocm only for a specific GPU architecture. Otherwise install it for all architectures.
if [[ -n ${APCI_AMD_GPU_ARCH+x} ]]; then
    ROCM_PACKAGE_VERSION="${APCI_HIP}-${APCI_AMD_GPU_ARCH}"
else
    ROCM_PACKAGE_VERSION="${APCI_HIP}"
fi

# TODO: It is not the minimal installation. There are many libraries, like fft and dnn are installed, which do not require.
quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    "amdrocm-core-dev${ROCM_PACKAGE_VERSION}"

unset ROCM_PACKAGE_VERSION
