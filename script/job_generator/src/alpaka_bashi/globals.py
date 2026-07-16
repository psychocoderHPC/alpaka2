"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

This module contains constants used for the alpaka job generation.
"""

import bashi
import packaging.version

# possible values of BUILD_TYPE
BUILD_TYPE: bashi.Parameter = "build_type"
CMAKE_RELEASE: int = 0
CMAKE_DEBUG: int = 1
CMAKE_RELEASE_WITH_DEBUG_INFO: int = 2
CMAKE_RELEASE_VER: bashi.ValueVersion = packaging.version.parse(str(CMAKE_RELEASE))
CMAKE_DEBUG_VER: bashi.ValueVersion = packaging.version.parse(str(CMAKE_DEBUG))
CMAKE_RELEASE_WITH_DEBUG_INFO_VER: bashi.ValueVersion = packaging.version.parse(str(CMAKE_RELEASE_WITH_DEBUG_INFO))
BUILD_TYPES: list[str | int | float] = [
    CMAKE_RELEASE,
    CMAKE_DEBUG,
    CMAKE_RELEASE_WITH_DEBUG_INFO,
]
BUILD_TYPES_NAMES: dict[str, bashi.ValueVersion] = {
    "Release": CMAKE_RELEASE_VER,
    "Debug": CMAKE_DEBUG_VER,
    "RelWithDebInfo": CMAKE_RELEASE_WITH_DEBUG_INFO_VER,
}
