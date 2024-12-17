/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#define ALPAKA_PP_CAT(left, right) left##right
#define ALPAKA_PP_REMOVE_FIRST_COMMA_DO(ignore, ...) __VA_ARGS__
#define ALPAKA_PP_REMOVE_FIRST_COMMA(...) ALPAKA_PP_REMOVE_FIRST_COMMA_DO(0 __VA_ARGS__)
#define ALPAKA_PP_REMOVE_BRACKETS(...) __VA_ARGS__
