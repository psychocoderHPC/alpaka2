"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the tags of the GitLab CI test job yaml.
"""

from typing import Any

import bashi
from typeguard import typechecked


# pylint: disable=unused-argument
@typechecked
def set_tags(job_body: dict[str, Any], combination: bashi.Combination):
    """Set the tags of the GitLab CI test job yaml depending on the combination.
    The tags decide which CI runner is used, e.g. the CPU runner for compile only jobs
    or the Nvidia runner for CUDA runtime jobs

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
