#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"
# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" != "Linux" ]]; then
    exit_error "Install OneAPI script does not support Windows or MacOS"
fi

: "${APCI_ONEAPI?'The OneAPI version must be specified'}"
: "${APCI_ONEAPI_TARGET?'The OneAPI target must be specified (cpu, intel_gpu, none)'}"

script_msg "Install OneAPI"

if [[ "$APCI_ONEAPI" != 0 ]]; then

    if agc-manager -e "oneapi@${APCI_ONEAPI}"; then
        echo_green "oneapi@${APCI_ONEAPI}"
        ONEAPI_PATH=$(agc-manager -b "oneapi@${APCI_ONEAPI}")
    else
        if [[ "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
            install_msg "OneAPI ${APCI_ONEAPI} in official APCI_ONEAPI container."
            ONEAPI_PATH=/opt/intel/oneapi
        else
            wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
                gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
            echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |
                sudo tee /etc/apt/sources.list.d/oneAPI.list
            retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

            quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
                intel-oneapi-common-vars \
                intel-oneapi-runtime-libs \
                intel-oneapi-compiler-dpcpp-cpp-"${APCI_ONEAPI}"

            ONEAPI_PATH=/opt/intel/oneapi
        fi
    fi

    # shellcheck source=script/ci/install/oneapi/setvars.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"

    APCI_C_COMPILER="${ONEAPI_PATH}/compiler/${APCI_ONEAPI}/bin/icx"
    APCI_CXX_COMPILER="${ONEAPI_PATH}/compiler/${APCI_ONEAPI}/bin/icpx"

    echo_green "${APCI_C_COMPILER} --version"
    $APCI_C_COMPILER --version
    echo_green "${APCI_CXX_COMPILER} --version"
    $APCI_CXX_COMPILER --version

    export ONEAPI_PATH
    export APCI_C_COMPILER
    export APCI_CXX_COMPILER
    store_variable ONEAPI_PATH
    store_variable APCI_C_COMPILER
    store_variable APCI_CXX_COMPILER
fi
