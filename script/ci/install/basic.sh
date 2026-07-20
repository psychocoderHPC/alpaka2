#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# Basic tools which are used from different dependencies

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/setup_vars.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/setup_vars.sh"
# shellcheck source=script/ci/utils/set.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/set.sh"
# shellcheck source=script/ci/utils/color_echo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/color_echo.sh"
# shellcheck source=script/ci/utils/misc.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/misc.sh"

########################
# install sudo
########################

# shellcheck source=script/ci/install/basic/sudo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/basic/sudo.sh"

########################
# install agc-manager
########################

# shellcheck source=script/ci/install/basic/agc-manager.sh
source "${APCI_ALPAKA_ROOT}/script/ci/install/basic/agc-manager.sh"

########################
# install via apt
########################

lazy_apt_update
# software-properties-common: 'add-apt-repository' and certificates for wget https download
# gnupg2 to add apt keys
# git for CMake fetch content
# calls cmake configure as subprocess and does not respect the generator of the parent cmake
# ninja for cmake build
_basic_apps=(software-properties-common wget gnupg2 git ninja-build)
quiet_run sudo DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y "${_basic_apps[@]}"
unset _basic_apps
