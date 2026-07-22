"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Write combination to stdout or file
"""

import argparse
from itertools import chain
from typing import Any, TextIO

import bashi
import yaml
from typeguard import typechecked

from alpaka_bashi.jobs import get_dummy_job_yaml, get_job_configuration


@typechecked
def write_job_yaml(
    jobs: dict[str, Any],
    output_stream: TextIO,
):
    """Write Python data structure to yaml output.

    Args:
        jobs (Dict[str, Any]): GitLab CI jobs.
        output_stream (TextIO): Python output stream where yaml is written to. For example stdout or
        a file handle.
    """
    for key, body in jobs.items():
        yaml.dump({key: body}, output_stream)
        output_stream.write("\n")


def write_single_file_job_configuration(
    pipelines: dict[bashi.ValueVersion, bashi.CombinationList],
    args: argparse.Namespace,
    output_stream: TextIO,
):
    """Write generated GitLab CI yaml code to stdout.

    Args:
        pipelines (dict[bashi.ValueVersion, bashi.CombinationList]): All CI pipelines and their
            jobs.
        args (argparse.Namespace): Application arguments.
    """

    jobs = get_job_configuration(
        combination_list=list(chain(*pipelines.values())),
        container_version=str(args.version),
        image_check=args.no_image_check,
        stages=False,
        wave_sizes=None,
    )

    if len(jobs) == 0:
        jobs = get_dummy_job_yaml()

    write_job_yaml(jobs, output_stream)
