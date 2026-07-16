"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import bashi


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

        return True
