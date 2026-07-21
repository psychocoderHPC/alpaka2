"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import bashi

from alpaka_bashi.versions import get_allowed_backend_combinations, get_used_backends


def check_only_valid_backend_combinations_a1(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Check if still possible valid backend combinations exist.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if (
        len(bashi.get_valid_compiler_backend_combinations(row, get_allowed_backend_combinations(), get_used_backends()))
        == 0
    ):
        alpaka_filter.reason("No valid backend combination available.")
        return False
    return True


# pylint: disable=too-few-public-methods
class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __call__(
        self,
        row: bashi.BashiRow,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (bashi.BashiRow): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """

        return check_only_valid_backend_combinations_a1(row, self)
