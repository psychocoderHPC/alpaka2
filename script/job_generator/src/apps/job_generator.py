"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generates the GitLab CI jobs for alpaka.
"""

import bashi

import alpaka_bashi


def main() -> None:
    """The main entry point."""
    software_versions = alpaka_bashi.get_software_versions_for_alpaka()
    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=software_versions, backends=alpaka_bashi.get_backends()
    )

    print(param_matrix)

    for name, values in param_matrix.items():
        print(name)
        for value in values:
            if name in ("host_compiler", "device_compiler"):
                print(f" {value.name}@{value.version}")

            else:
                print(f" {value.version}")


if __name__ == "__main__":
    main()
