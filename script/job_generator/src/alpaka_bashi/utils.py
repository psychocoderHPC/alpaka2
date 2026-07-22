"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Utils for the job-generator
"""

import termcolor
from typeguard import typechecked


@typechecked
def print_warn(msg: str):
    """Print message in yellow with a [WARNING] prefix.

    Args:
        msg (str): warning text
    """
    print(termcolor.colored(f"[WARNING]: {msg}", "yellow"))
