"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Different GitLab CI yaml code snippets, which are not in an extra file.
"""

from typing import Any

from typeguard import typechecked


@typechecked
def set_misc_job_properties(job_body: dict[str, Any]):
    """Set different GitLab CI job yaml properties.

    Args:
        job_body (Dict[str, Any]): _description_
    """
    job_body["interruptible"] = True
