#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

echo_green "ROCm install method up to version 7.2"

sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - |
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null

# Prevents apt warnings when the script is run a second time.
# Delete and recreate the source list to ensure that the correct apt sources are set.
if [[ -f /etc/apt/sources.list.d/rocm.list ]]; then
    sudo rm -rf /etc/apt/sources.list.d/rocm.list
fi

# require to set environment variable VERSION_CODENAME
source /etc/os-release
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${APCI_HIP} ${VERSION_CODENAME} main" |
    sudo tee -a /etc/apt/sources.list.d/rocm.list
if [ "$(version "${APCI_HIP}")" -ge "$(version "7")" ]; then
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/${APCI_HIP}/ubuntu ${VERSION_CODENAME} main" |
        sudo tee -a /etc/apt/sources.list.d/rocm.list
fi

retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

APCI_ROCM="${APCI_HIP}"
# append .0 if no patch level is defined
if ! echo "${APCI_ROCM}" | grep -Eq '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+'; then
    APCI_ROCM="${APCI_ROCM}.0"
fi

quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    "rocm-llvm${APCI_ROCM}" \
    "hip-runtime-amd${APCI_ROCM}" \
    "rocm-dev${APCI_ROCM}" \
    "rocm-utils${APCI_ROCM}" \
    "rocrand-dev${APCI_ROCM}" \
    "rocminfo${APCI_ROCM}" \
    "rocm-cmake${APCI_ROCM}" \
    "rocm-device-libs${APCI_ROCM}" \
    "rocm-core${APCI_ROCM}" \
    "rocm-smi-lib${APCI_ROCM}"
