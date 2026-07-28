"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generate a dummy job, if GitLab CI yaml file would be empty.
"""

from typeguard import typechecked


@typechecked
def get_dummy_job(stage: str = "") -> dict:
    """Return GitLab CI job, which simply prints a message. Can be used, if no job is generated for
    a CI pipeline.

    Args:
        stage (str, optional): Set stage, if string is not empty. Defaults to "".
    """

    job_name = "dummy-job"

    job = {
        job_name: {
            "image": "alpine:latest",
            "interruptible": True,
            "script": ['echo "This is a dummy job so that the CI does not fail."'],
        }
    }

    if stage:
        job[job_name]["stage"] = stage

    return job
