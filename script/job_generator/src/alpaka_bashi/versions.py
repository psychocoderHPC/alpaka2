"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Software versions to be tested.
"""

from copy import deepcopy

import packaging.version
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    CLANG,
    CLANG_CUDA,
    CMAKE,
    CXX_STANDARD,
    GCC,
    HIPCC,
    ICPX,
    NVCC,
    UBUNTU,
)
from bashi.version.dependencies.clang_cuda import CLANG_CUDA_MAX_CUDA_VERSION

from alpaka_bashi.globals import BUILD_TYPE, BUILD_TYPES

ALPAKA_VERSIONS: dict[str, list[str | int | float]] = {
    GCC: [12, 13, 14, 15],
    CLANG: [17, 18, 19, 20, 21],
    NVCC: [12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3],
    HIPCC: [6.2, 6.3, 6.4, 7.0, 7.1, 7.2],
    ICPX: ["2025.1", "2025.2", "2025.3", "2026.0"],
    UBUNTU: ["22.04", "24.04"],
    CMAKE: ["3.25.3", "3.26.4", "3.27.9", "3.28.6", "3.29.8", "3.30.3"],
    CXX_STANDARD: ["20"],
    BUILD_TYPE: BUILD_TYPES,
}


def _get_clang_cuda_versions() -> list[str | int | float]:
    """Return a list of Clang-CUDA versions. If there is no CUDA version
    bashi.versions.CLANG_CUDA_MAX_CUDA_VERSION which supports a specific Clang-CUDA, don't it add to
    the list.

    Returns:
        List[Union[str, int, float]]: List of Clang-CUDA versions.
    """
    min_cuda_version = packaging.version.parse(str(min(ALPAKA_VERSIONS[NVCC])))
    min_clang_cuda_version = packaging.version.parse("0")
    for clang_cuda_sdk in sorted(CLANG_CUDA_MAX_CUDA_VERSION):
        if min_cuda_version <= clang_cuda_sdk.cuda:
            min_clang_cuda_version = clang_cuda_sdk.clang_cuda
            break
    return [ver for ver in ALPAKA_VERSIONS[CLANG] if packaging.version.parse(str(ver)) >= min_clang_cuda_version]


def get_software_versions_for_alpaka() -> dict[str, list[str | int | float]]:
    """Return dict of all compiler and software versions, which should be used as input for the
    combination generator.

    Raises:
        RuntimeError: If no valid Clang-CUDA versions exist.

    Returns:
        Dict[str, List[Union[str, int, float]]]: List of compiler and software versions.
    """

    clang_cuda_versions = _get_clang_cuda_versions()
    # The alpaka filter function cannot handle the case, that Clang-CUDA compiler are missing.
    # In the case, that the parameter-value-matrix is missing Clang-CUDA, we get a meaning full
    # error.
    if len(clang_cuda_versions) == 0:
        raise RuntimeError("Alpaka custom filter does not work without Clang-CUDA version.")

    return deepcopy(ALPAKA_VERSIONS) | {CLANG_CUDA: clang_cuda_versions}


def get_backends() -> list[str]:
    """Return the list of backends, used by alpaka."""
    return [
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        ALPAKA_ACC_ONEAPI_CPU_ENABLE,
        ALPAKA_ACC_ONEAPI_GPU_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
        ALPAKA_ACC_GPU_HIP_ENABLE,
    ]
