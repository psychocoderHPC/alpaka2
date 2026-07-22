"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Set the script section of the GitLab CI test job yaml.
"""

from typing import Any

from typeguard import typechecked


@typechecked
def set_script(job_body: dict[str, Any]):
    """Set the job section of a job. Overwrite an existing job section."""
    job_body["script"] = [
        "$APCI_ALPAKA_ROOT/script/ci/info.sh",
        "$APCI_ALPAKA_ROOT/script/ci/install.sh",
        "$APCI_ALPAKA_ROOT/script/ci/configure.sh",
        "$APCI_ALPAKA_ROOT/script/ci/build.sh",
        "$APCI_ALPAKA_ROOT/script/ci/test.sh",
    ]
