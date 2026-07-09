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

parse_compiler_version "$APCI_DEVICE_COMPILER"

# if we want to compile alpaka with the icpx compiler and the tbb backend, the OneAPI SDK is also required
if [[ "$APCI_ONEAPI" != 0 || "$compiler_name" == "icpx" ]]; then
    if [[ "$APCI_ONEAPI" != 0 ]]; then
        one_api_version="$APCI_ONEAPI"
    else
        one_api_version="$compiler_version"
    fi

    if agc-manager -e "oneapi@${one_api_version}"; then
        echo_green "oneapi@${one_api_version}"
        ONEAPI_PATH=$(agc-manager -b "oneapi@${one_api_version}")
    else
        if [[ "${APCI_IMAGE_NAME}" =~ "intel/oneapi" ]]; then
            install_msg "use OneAPI ${one_api_version} from official OneAPI container."
            ONEAPI_PATH=/opt/intel/oneapi
        else
            wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
                gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null
            echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |
                sudo tee /etc/apt/sources.list.d/oneAPI.list
            retry_cmd sudo DEBIAN_FRONTEND=noninteractive apt update

            quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
                "$(apt_package_version intel-oneapi-runtime-opencl "${one_api_version}")" \
                intel-oneapi-compiler-dpcpp-cpp-"${one_api_version}"

            ONEAPI_PATH=/opt/intel/oneapi
        fi
    fi

    # shellcheck source=script/ci/install/oneapi/setvars.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/install/oneapi/setvars.sh"

    echo "Search for icx and icpx with 'which'"
    # Use which to search for the compiler. If the /opt/intel/oneapi/setvars.sh script
    # does not work, which will not find anything and the script exit because of return
    # code non 0.
    APCI_C_COMPILER="$(which icx)"
    APCI_CXX_COMPILER="$(which icpx)"

    echo_run "$APCI_C_COMPILER" --version
    echo_run "$APCI_CXX_COMPILER" --version

    echo_run sycl-ls

    export ONEAPI_PATH
    export APCI_C_COMPILER
    export APCI_CXX_COMPILER
    store_variable ONEAPI_PATH
    store_variable APCI_C_COMPILER
    store_variable APCI_CXX_COMPILER
else
    echo_green "Skipped install OneAPI because it is not required for the job."
fi
