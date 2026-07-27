"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from collections.abc import Callable

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    CLANG,
    DEVICE_COMPILER,
    GCC,
    HOST_COMPILER,
    NVCC,
    OFF,
    ON,
)

from alpaka_bashi.versions import get_allowed_backend_combinations, get_used_backends, get_used_compiler_versions


def remove_disabled_serial_backend_for_gcc_and_clang(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """GCC and Clang as device compiler uses the serial backend all the time."""
    for compiler_name in (GCC, CLANG):
        bashi.remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=DEVICE_COMPILER,
            value_name1=compiler_name,
            parameter2=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            value_version2=OFF,
        )


def remove_disabled_serial_and_openmp_backend(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """The serial backend is tested with the openmp backend all the time."""

    bashi.remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        value_version1=OFF,
        parameter2=ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        value_version2=ON,
    )


def remove_nvcc_clang_combinations(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """Remove all parameter-value-pairs which are related to the combinations nvcc + clang host
    compiler."""
    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
    )

    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        value_max_version2=OFF,
    )


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

    remove_disabled_serial_backend_for_gcc_and_clang(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_disabled_serial_and_openmp_backend(expected_param_val_tuple, unexpected_param_val_tuple)
    # TODO: Remove me, if nvcc + clang is enabled
    remove_nvcc_clang_combinations(expected_param_val_tuple, unexpected_param_val_tuple)

    expected_param_val_okay = bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )
    unexpected_param_val_okay = bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )

    return expected_param_val_okay and unexpected_param_val_okay
