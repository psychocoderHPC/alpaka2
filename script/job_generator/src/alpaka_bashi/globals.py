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


def get_version_aliases() -> dict[bashi.ValueName, dict[bashi.ValueVersion, str]]:
    """Return a list of value-version aliases which can be set for print_row_nice()

    Returns:
        Dict[bashi.ValueName, Dict[bashi.ValueVersion, str]]: _description_
    """
    version_aliases = {}
    for val_name, version_map in [
        (BUILD_TYPE, BUILD_TYPES_NAMES),
    ]:
        version_map_parsed: dict[bashi.ValueVersion, str] = {}
        for alias, ver in version_map.items():
            version_map_parsed[ver] = alias
        version_aliases[val_name] = version_map_parsed

    return version_aliases
