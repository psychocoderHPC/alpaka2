"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

alpaka_bashi package
"""

from alpaka_bashi.alpaka_filter import AlpakaFilter
from alpaka_bashi.combination import add_combinations_parameters
from alpaka_bashi.globals import (
    BUILD_TYPE,
    BUILD_TYPES,
    BUILD_TYPES_NAMES,
    CMAKE_DEBUG,
    CMAKE_DEBUG_VER,
    CMAKE_RELEASE,
    CMAKE_RELEASE_VER,
    CMAKE_RELEASE_WITH_DEBUG_INFO,
    CMAKE_RELEASE_WITH_DEBUG_INFO_VER,
    HWLOC,
    get_version_aliases,
)
from alpaka_bashi.verify import verify
from alpaka_bashi.versions import (
    get_alpaka_version_relation,
    get_software_versions_for_alpaka,
    get_used_backends,
)

__all__ = [
    "AlpakaFilter",
    "add_combinations_parameters",
    "BUILD_TYPE",
    "BUILD_TYPES",
    "BUILD_TYPES_NAMES",
    "CMAKE_DEBUG",
    "CMAKE_DEBUG_VER",
    "CMAKE_RELEASE",
    "CMAKE_RELEASE_VER",
    "CMAKE_RELEASE_WITH_DEBUG_INFO",
    "CMAKE_RELEASE_WITH_DEBUG_INFO_VER",
    "HWLOC",
    "verify",
    "get_version_aliases",
    "get_alpaka_version_relation",
    "get_used_backends",
    "get_software_versions_for_alpaka",
]
