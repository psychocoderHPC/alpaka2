"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Check if a given combination of parameter-values is supported by alpaka.
"""

import sys

import bashi
import bashiValidate
import termcolor
from bashi.globals import OFF, ON

import alpaka_bashi


def main():
    """The main entry point."""
    validator = bashiValidate.Validator(version_relation=alpaka_bashi.get_alpaka_version_relation())
    validator.parser.add_argument(
        "--missing-parameters",
        action="store_true",
        help="Display all missing parameters not set via application argument",
    )
    validator.add_custom_filter(alpaka_bashi.AlpakaFilter())
    validator.add_string_parameter(
        alpaka_bashi.BUILD_TYPE, "CMake build type.", alpaka_bashi.BUILD_TYPES_NAMES, "buildType"
    )
    validator.add_known_version(alpaka_bashi.BUILD_TYPE, [str(option) for option in alpaka_bashi.BUILD_TYPES])
    validator.add_software_version_parameter(
        alpaka_bashi.HWLOC, "Enable hwloc and therefore Numa support.", choices=["ON", "OFF"]
    )
    validator.add_known_version(alpaka_bashi.HWLOC, [OFF, ON])

    arg = validator.parser.parse_args()
    if arg.missing_parameters:
        row = validator.get_row()
        missing_parameter: list[str] = [
            parameter
            for parameter in bashi.get_parameter_value_matrix(
                software_versions=alpaka_bashi.get_software_versions_for_alpaka()
            )
            if parameter not in row
        ]

        if len(missing_parameter) > 0:
            print(
                termcolor.colored(
                    f"Missing parameter: {', '.join(missing_parameter)}",
                    "yellow",
                )
            )

    sys.exit(int(not validator.validate()))


if __name__ == "__main__":
    main()
