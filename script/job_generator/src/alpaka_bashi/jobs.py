"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate GitLab CI jobs for a given combination
"""

import re
from typing import Any

import bashi
import packaging.version
from bashi.globals import CLANG, GCC
from typeguard import typechecked

from alpaka_bashi.ci_yaml.misc import get_dummy_job
from alpaka_bashi.ci_yaml.names import get_job_name
from alpaka_bashi.globals import CI_PIPELINE_NAME, get_version_aliases
from alpaka_bashi.jobs_builder.default import construct_job_yaml
from alpaka_bashi.jobs_builder.emulated_simd import get_emulated_simd_job
from alpaka_bashi.jobs_builder.santizer import SanitizerType, get_sanitizer_job
from alpaka_bashi.versions import get_used_compiler_versions


@typechecked
def get_job_configuration(
    combination_list: bashi.CombinationList,
    container_version: str,
    image_check: bool,
    stages: bool,
    wave_sizes: dict[bashi.ValueVersion, int] | None = None,
) -> dict[str, Any]:
    """Generate for each combination a GitLab CI yaml.

    Args:
        combination_list (bashi.CombinationList): combination-list
        container_version (str): Alpaka CI container tag.
        image_check (bool): If true, check if alpaka CI image exist (requires internet connection).
        stages (bool): If true, add stages.
        wave_sizes (Dict[ValueVersion, int] | None, optional): The wave size defines how many jobs
        can be in one stage of a CI pipeline. The key defines the pipeline and value maximum number
        of jobs in a CI stage. If a pipeline is not defined in the dict, put all jobs in the same
        stage. Defaults to None.

    Returns:
        Dict[str, Any]: GitLab CI job yaml's
    """
    jobs: dict[str, Any] = {}

    if len(combination_list) > 0 and stages:
        jobs["stages"] = []

    stage_job_counter: dict[bashi.ValueVersion, int] = {}
    if wave_sizes is not None:
        for wave_ver in wave_sizes:
            stage_job_counter[wave_ver] = 0

    for comb in combination_list:
        job_name = get_job_name(comb)
        wave_ver = comb[CI_PIPELINE_NAME].version

        if stages:
            stage_name = get_version_aliases()[CI_PIPELINE_NAME][wave_ver]

            if wave_sizes is not None and wave_ver in stage_job_counter:
                # dived number of already generated jobs by the wave size and round down.
                stage_name += f"_stage{int(stage_job_counter[wave_ver] / wave_sizes[wave_ver])}"
                stage_job_counter[wave_ver] += 1

            if stage_name not in jobs["stages"]:
                jobs["stages"].append(stage_name)
        else:
            stage_name = ""

        jobs[job_name] = construct_job_yaml(comb, stage_name, container_version, image_check)

    return jobs


@typechecked
def get_special_jobs(
    container_version: str,
    image_check: bool,
    stage_name: str,
    job_filter: str,
) -> dict[str, Any]:
    """Return Dict of special CI jobs.

    Args:
        container_version (str): Container version.
        image_check (bool): Check if configured image exist. If not, use fallback.
        stage_name (str): Stage name. If empty, do not create stage property.
        job_filter (str): Filter jobs by job name. If empty, do not filter.

    Returns:
        Dict[str, Any]: Dict of CI jobs.
    """
    special_jobs: dict[str, Any] = {}

    if stage_name:
        special_jobs["stages"] = [stage_name]

    for compiler in (GCC, CLANG):
        for sanitzer in SanitizerType:
            special_jobs |= get_sanitizer_job(
                compiler_name=compiler,
                compiler_version=packaging.version.parse(str(max(get_used_compiler_versions()[compiler]))),
                sanitizer_type=sanitzer,
                container_version=container_version,
                stage_name=stage_name,
                image_check=image_check,
            )

    for compiler in (GCC, CLANG):
        special_jobs |= get_emulated_simd_job(
            compiler_name=compiler,
            compiler_version=packaging.version.parse(str(max(get_used_compiler_versions()[compiler]))),
            container_version=container_version,
            stage_name=stage_name,
            image_check=image_check,
        )

    if job_filter:
        compiled_regex = re.compile(job_filter)
        special_jobs = {
            job_name: job_body
            for job_name, job_body in special_jobs.items()
            if compiled_regex.match(job_name) or job_name == "stages"
        }

    return special_jobs


@typechecked
def get_dummy_job_yaml(stage_name: str = "") -> dict[str, Any]:
    """Generate a dummy job, which can never fail.

    Args:
        stage_name (str, optional): Set stage, if string is not empty. Defaults to "".

    Returns:
        Dict[str, Any]: CI job yaml.
    """
    dummy_job: dict[str, Any] = {}
    if stage_name != "":
        dummy_job["stages"] = [stage_name]

    dummy_job |= get_dummy_job()
    if stage_name != "":
        dummy_job["dummy-job"]["stage"] = stage_name
    return dummy_job
