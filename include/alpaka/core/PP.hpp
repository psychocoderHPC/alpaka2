/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#define ALPAKA_PP_CAT(left, right) left##right
#define ALPAKA_PP_REMOVE_FIRST_COMMA_DO(ignore, ...) __VA_ARGS__
#define ALPAKA_PP_REMOVE_FIRST_COMMA(...) ALPAKA_PP_REMOVE_FIRST_COMMA_DO(0 __VA_ARGS__)

/** solution from https://stackoverflow.com/a/62984543
 * @{
 */
#define ALPAKA_PP_REMOVE_BRACKETS_DO(X) ALPAKAESC(ISHALPAKA X)
#define ISHALPAKA(...) ISHALPAKA __VA_ARGS__
#define ALPAKAESC(...) ALPAKAESC_(__VA_ARGS__)
#define ALPAKAESC_(...) VAN##__VA_ARGS__
#define VANISHALPAKA
/** @} */

#define ALPAKA_PP_REMOVE_BRACKETS(x) ALPAKA_PP_REMOVE_BRACKETS_DO(x)
