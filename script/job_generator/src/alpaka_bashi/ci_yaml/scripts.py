"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the script section of the GitLab CI test job yaml.
"""

from typing import Any

from typeguard import typechecked


# pylint: disable=unused-argument
@typechecked
def set_script(job_body: dict[str, Any]):
    """Set the job section of a job. Overwrite an existing job section."""
