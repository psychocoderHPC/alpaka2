#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# print all job variables and depending on the OS, how to run the job locally.

# shellcheck source=script/ci/utils/default.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/default.sh"

if [[ "$APCI_OS_NAME" == "Linux" ]]; then
    # shellcheck source=script/ci/info/linux.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/info/linux.sh"
else
    # shellcheck source=script/ci/info/windows_macos.sh
    source "${APCI_ALPAKA_ROOT}/script/ci/info/windows_macos.sh"
fi
