"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generates sanitizer jobs for the CI.
"""

# The linter error is triggered by manually defining a combination
# pylint: disable=duplicate-code

from enum import StrEnum
from typing import Any

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    ALPAKA_ACC_GPU_HIP_ENABLE,
    ALPAKA_ACC_ONEAPI_CPU_ENABLE,
    ALPAKA_ACC_ONEAPI_GPU_ENABLE,
    CMAKE,
    CXX_STANDARD,
    DEVICE_COMPILER,
    HOST_COMPILER,
    OFF,
    ON,
    UBUNTU,
)

from alpaka_bashi.globals import (
    BUILD_TYPE,
    CI_PIPELINE_NAME,
    CI_PIPELINE_SPECIAL_VER,
    CMAKE_DEBUG,
    HWLOC,
    JOB_EXECUTION_RUNTIME,
    JOB_EXECUTION_TYPE,
)
from alpaka_bashi.jobs_builder.default import construct_job_yaml


class SanitizerType(StrEnum):
    """Sanitizer Type"""

    ASAN = "APCI_SANITIZER_ASAN"
    TSAN = "APCI_SANITIZER_TSAN"
    LSAN = "APCI_SANITIZER_LSAN"
    UBSAN = "APCI_SANITIZER_UBSAN"


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def get_sanitizer_job(
    compiler_name: bashi.ValueName,
    compiler_version: bashi.ValueVersion,
    sanitizer_type: SanitizerType,
    container_version: str,
    stage_name: str,
    image_check: bool,
) -> dict[str, Any]:
    """
    Create a GitLab CI sanitizer job

    Args:
        compiler_name (bashi.ValueName): Name of the used compiler.
        compiler_version (bashi.ValueVersion): Name of the used compiler.
        sanitizer_type (SanitizerType): Type of the sanitizer.
        container_version (str): Container version
        stage_name (str): Stage name. If empty do not set an stage property.
        image_check (bool): Check if image exist. If not, use fallback image.

    Returns:
        dict[str, Any]: Description.
    """
    job_body = construct_job_yaml(
        combination=bashi.parse_combination(
            [
                (HOST_COMPILER, compiler_name, compiler_version),
                (DEVICE_COMPILER, compiler_name, compiler_version),
                (CMAKE, "3.30.3"),
                (UBUNTU, "24.04"),
                (CXX_STANDARD, 20),
                (BUILD_TYPE, CMAKE_DEBUG),
                (JOB_EXECUTION_TYPE, JOB_EXECUTION_RUNTIME),
                (CI_PIPELINE_NAME, CI_PIPELINE_SPECIAL_VER),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, OFF),
                (ALPAKA_ACC_GPU_HIP_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_CPU_ENABLE, OFF),
                (ALPAKA_ACC_ONEAPI_GPU_ENABLE, OFF),
                (HWLOC, ON),
            ]
        ),
        stage=stage_name,
        container_version=container_version,
        image_check=image_check,
    )
    job_body["variables"][sanitizer_type.value] = "ON"

    name_prefix = sanitizer_type.value.split("_")[-1].lower()

    return {f"linux_{compiler_name}_{compiler_version}_{name_prefix}": job_body}
