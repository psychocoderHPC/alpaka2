"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Utils for the job-generator
"""

import argparse
import os
import sys

import termcolor
from typeguard import typechecked


@typechecked
def print_warn(msg: str):
    """Print message in yellow with a [WARNING] prefix.

    Args:
        msg (str): warning text
    """
    print(termcolor.colored(f"[WARNING]: {msg}", "yellow"), file=sys.stderr)


def get_filter_name(args: argparse.Namespace) -> str:
    """Return filter string CI jobs. All jobs, which does not match the filter regex, will be
    removed.

    Ether the filter is set via command line argument --filter or via Git commit message with the
    prefix `CI_FILTER:`.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        str: The filter regex. Return empty string, if no filter was set.
    """
    commit_message_filter_prefix = "CI_FILTER:"
    if os.getenv("CI_COMMIT_MESSAGE"):
        for line in os.getenv("CI_COMMIT_MESSAGE", "").split("\n"):
            striped_line = line.strip()
            if striped_line.strip().startswith(commit_message_filter_prefix):
                return striped_line[len(commit_message_filter_prefix) :].strip()

    if args.filter:
        return args.filter

    return ""
