"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Distribute Jobs to pipelines and reorder it.
"""

import bashi

from alpaka_bashi.globals import CI_PIPELINE_NAME, CI_PIPELINE_NAME_MAPPING


def distribute_to_pipelines(
    combinations: bashi.CombinationList,
) -> dict[bashi.ValueVersion, bashi.CombinationList]:
    """Distribute the combinations depending on the CI_PIPELINE_NAME value-version to different CI
    pipelines.

    Args:
        combinations (bashi.CombinationList): combination-list

    Returns:
        Dict[bashi.ValueVersion, bashi.CombinationList]: The key is the name of the CI pipeline, the
        values are all jobs of it.
    """
    waves: dict[bashi.ValueVersion, bashi.CombinationList] = {}
    for wave in CI_PIPELINE_NAME_MAPPING.values():
        waves[wave] = []

    for comb in combinations:
        waves[comb[CI_PIPELINE_NAME].version].append(comb)

    return waves
