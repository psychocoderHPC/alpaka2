#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${ONEAPI_PATH?'ONEAPI_PATH path is not set'}"

# the OneAPI setvars.sh script has unbound variable and does not catch every non zero return value
unset_set_e=false
if echo "$-" | grep -q 'e'; then
    set +e
    unset_set_e=true
fi

unset_set_u=false
if echo "$-" | grep -q 'u'; then
    set +u
    unset_set_u=true
fi

# shellcheck source=/dev/null
source "${ONEAPI_PATH}/setvars.sh"

if [[ $unset_set_e == true ]]; then
    set -e
    unset_set_e=false
fi

if [[ $unset_set_u == true ]]; then
    set -u
    unset_set_u=false
fi
