"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from typing import Callable

import bashi

from alpaka_bashi.versions import get_allowed_backend_combinations, get_used_backends, get_used_compiler_versions


def verify(
    combination_list: bashi.CombinationList,
    param_value_matrix: bashi.ParameterValueMatrix,
    version_relation: bashi.VersionRelation,
    run_infos: dict[str, Callable[..., bool]],
) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """

    expected_param_val_tuple, unexpected_param_val_tuple = bashi.get_expected_bashi_parameter_value_pairs(
        param_value_matrix, version_relation, run_infos
    )

    bashi.remove_unsupported_compiler_backend_combinations(
        expected_param_val_tuple,
        unexpected_param_val_tuple,
        list(get_used_compiler_versions().keys()),
        get_used_backends(),
        get_allowed_backend_combinations(),
    )
    bashi.remove_unsupported_backend_combinations(
        expected_param_val_tuple,
        unexpected_param_val_tuple,
        get_used_backends(),
        get_allowed_backend_combinations(),
    )

    expected_param_val_okay = bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )
    unexpected_param_val_okay = bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )

    return expected_param_val_okay and unexpected_param_val_okay
