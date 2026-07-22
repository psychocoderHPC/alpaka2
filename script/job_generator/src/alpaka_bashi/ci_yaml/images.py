"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the image of the GitLab CI test job yaml.
"""

from typing import Any

import bashi
from typeguard import typechecked


# pylint: disable=unused-argument
@typechecked
def set_image(
    job_body: dict[str, Any],
    combination: bashi.Combination,
    container_version: str,
    image_check: bool,
):
    """Set the image of the GitLab CI test job yaml depending on the combination.

    Args:
        job_body (Dict[str, Any]): GitLab CI test job body yaml
        combination (bashi.Combination): combination
    """
