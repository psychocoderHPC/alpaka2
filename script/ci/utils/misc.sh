#!/usr/bin/env bash

#
# Copyright 2026 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

: "${APCI_ALPAKA_ROOT?'APCI_ALPAKA_ROOT is not defined. Root directory of the alpaka project'}"

# shellcheck source=script/ci/utils/color_echo.sh
source "${APCI_ALPAKA_ROOT}/script/ci/utils/color_echo.sh"

# display a install message in green
install_msg() {
    echo_green "[INSTALL]: " "$@"
}

# display a install message in green
script_msg() {
    echo_green "[SCRIPT]: " "$@"
}

# display a error message in red
error_msg() {
    echo_red "[ERROR]: " "$@"
}

# display bash stack trace
trace() {
    echo "TRACE: "
    for ((i = ${#FUNCNAME[@]} - 1; i; i--)); do
        printf '%*s%s() %s:%s\n' "$((${#FUNCNAME[@]} - "$i"))" '' \
            "${FUNCNAME[i]}" "${BASH_SOURCE[i]}" "${BASH_LINENO[i - 1]}"
    done
}

exit_error() {
    error_msg "$1"
    trace

    if [[ $# -lt 2 ]]; then
        exit 1
    else
        exit "$2"
    fi
}

# Parse compiler version string with the shape of compiler_name@compiler_version and
# store result in variables compiler_name and compiler_version.
#
# Example:
#
# parse_compiler_version gcc@15
# echo $compiler_name # output: gcc
# echo $compiler_version # output: 15
parse_compiler_version() {
    if ! echo "$1" | grep -q '@'; then
        exit_error "parse_compiler_version(): No @ in variable string '${1}'"
    fi

    IFS='@' read -r compiler_name compiler_version <<<"$1"

    export compiler_name
    export compiler_version
}

# parse a string containing a version number to a format, which can be compare in a if statement
#
# e.g.  if [ "$(version "${APCI_ROCM}")" -ge "$(version "6.0.0")" ]; then
#
# allowed version numbers:
# - major: 7
# - major.minor: 7.1
# - major-minor.patch: 7.1.2
# - major-minor.patch.subpatch: 7.1.2.8
function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

# If an error occurs (command does not return 0), try running the command again.
# Usage: retry_cmd command arg1 arg2 ...
#
# Configure variables
# - RETRY_CMD_MAX: number of retires (default 10)
# - RETRY_CMD_WAIT: wait N seconds between two tries (default 1)
retry_cmd() {
    if [[ $# -lt 1 ]]; then
        exit_error "retry_cmd requires at least one argument."
    fi

    echo_green "$*"
    (
        set +euo pipefail
        local max_tries="${RETRY_CMD_MAX:-10}"

        # time in seconds
        local wait_time="${RETRY_CMD_WAIT:-1}"

        for ((i = 0; i < max_tries; ++i)); do
            "$@"
            result="$?"

            if [[ "$result" -eq 0 ]]; then
                return 0
            fi

            echo_yellow "[WARNING]: Attempt #${i} to run '$*' failed"
            sleep "$wait_time"
        done
        exit_error "run '$*' failed" "$result"
    )
}

# display command and execute it
echo_run() {
    echo_green "$*"
    "$@"
}

# Run a command suppress the output if the error code is 0.
# If the error code is non 0, display output and run exit_error.
quiet_run() {
    if [[ $# -lt 1 ]]; then
        exit_error "quiet_run requires at least one argument."
    fi

    echo_green "$*"

    local set_options=$-
    exit_no_non_zero=$(
        echo "${set_options}" | grep -q -v 'e'
        echo "$?"
    )

    local log_file
    log_file=$(mktemp)

    # disable temporary exit on non zero, that log can be displayed
    if [[ ${exit_no_non_zero} -eq 1 ]]; then
        set +e
    fi

    "$@" >"${log_file}" 2>&1
    local exit_code=$?

    if [[ ${exit_no_non_zero} -eq 1 ]]; then
        set -e
    fi

    if [[ "${exit_code}" != 0 ]]; then
        cat "${log_file}"
    fi

    rm "${log_file}"

    if [[ ${exit_code} -ne 0 ]]; then
        exit_error "failed cmd: " "$@"
    fi
}

# does an 'apt update' only if no 'apt update' was done before
# ATTENTION: If you add a new ppa no 'apt update' is performed. Instead call directly
# `DEBIAN_FRONTEND=noninteractive apt update`.
lazy_apt_update() {
    if [[ -z "$(ls -A '/var/lib/apt/lists/')" ]]; then
        DEBIAN_FRONTEND=noninteractive retry_cmd apt update
    fi
}

# Print a string to install a specific version of an apt package.
#
# Args:
#   1. Name of the package
#   2. Version searching for
#
# If you want to install a specific version an apt package you have to
# set the exact version which can contain a patch level, distribution
# depending patch level and more. The function is searching for a
# apt version, which matches best the given version string.
#
# The printed value can be directly used in a `apt install` command.
apt_package_version() {
    if [[ $# -lt 1 ]]; then
        exit_error "apt_package_version(): no package name set."
    fi

    if [[ $# -lt 2 ]]; then
        exit_error "apt_package_version(): no package version set."
    fi

    local apt_ver
    # search for a specific version
    # if more than one apt version matches the given version, take the latest version
    apt_ver=$(apt list --all-versions "$1" |
        cut -d " " -f 2 |
        grep "$2" |
        sort -V | tail -n 1)

    if [[ "${apt_ver}" == "" ]]; then
        exit_error "apt_package_version(): Didn't find any apt version for package $1 with version $2 ."
    fi

    echo "$1=$apt_ver"
}

# Run wget with different output options, depending if it is executed locally or in the CI.
#
# Args:
#  - 1. Source URL
#  - 2. Target file location
ci_wget() {
    if [[ $# -lt 1 ]]; then
        exit_error "ci_wget(): 1. Argument is missing: No source is defined."
    fi

    if [[ $# -lt 2 ]]; then
        exit_error "ci_wget(): 2. Argument is missing: No target is defined."
    fi

    if [[ -n ${CI+x} ]]; then
        # be quite in the CI
        retry_cmd wget --quiet "$1" -O "$2"
    else
        retry_cmd wget "$1" -O "$2"
    fi
}

# print a variable in the format `name=value`, if the value is not empty
# usage: echo_if_not_empty <variable_name>
echo_if_not_empty() {
    if [[ $# -lt 1 ]]; then
        exit_error "echo_if_not_empty(): no variable name set."
    fi

    local var_name="$1"
    if [[ "${!var_name}" != "" ]]; then
        echo "${var_name}=${!var_name}"
    fi
}
