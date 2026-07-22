"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generates default jobs for the CI.
"""

from typing import Any

import bashi
from typeguard import typechecked

from alpaka_bashi.ci_yaml.images import set_image
from alpaka_bashi.ci_yaml.misc import set_misc_job_properties
from alpaka_bashi.ci_yaml.scripts import set_script
from alpaka_bashi.ci_yaml.tags import set_tags
from alpaka_bashi.ci_yaml.variables import set_variables


@typechecked
def construct_job_yaml(
    combination: bashi.Combination,
    stage: str,
    container_version: str,
    image_check: bool,
) -> dict[str, Any]:
    """Construct a GitLab CI test job body yaml from the given combination.

    Args:
        combination (bashi.Combination): combination
        stage (str): Name of the pipeline stage. If empty, do not create stages.
        container_version (str): Alpaka CI container tag.
        image_check (bool): If true, check if alpaka CI image exist (requires internet connection).

    Returns:
        Dict[str, Any]: GitLab CI job body
    """
    job_body = {}

    if stage:
        job_body["stage"] = stage
    set_image(job_body, combination, container_version, image_check)
    set_variables(job_body, combination)
    set_script(job_body)
    set_tags(job_body, combination)
    set_misc_job_properties(job_body)

    return job_body
