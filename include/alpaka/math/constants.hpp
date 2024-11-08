/* Copyright 2023 Benjamin Worpitz, Matthias Werner, Jan Stephan, Bernhard Manfred Gruber, Sergei Bastrakov,
 *                Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cmath>
#include <complex>
#if __has_include(<version>) // Not part of the C++17 standard but all major standard libraries include this
#    include <version>
#endif
#ifdef __cpp_lib_math_constants
#    include <numbers>
#endif

namespace alpaka::math
{
    namespace constants
    {
#ifdef __cpp_lib_math_constants
        inline constexpr double e = std::numbers::e;
        inline constexpr double log2e = std::numbers::log2e;
        inline constexpr double log10e = std::numbers::log10e;
        inline constexpr double pi = std::numbers::pi;
        inline constexpr double inv_pi = std::numbers::inv_pi;
        inline constexpr double ln2 = std::numbers::ln2;
        inline constexpr double ln10 = std::numbers::ln10;
        inline constexpr double sqrt2 = std::numbers::sqrt2;

        template<typename T>
        inline constexpr T e_v = std::numbers::e_v<T>;

        template<typename T>
        inline constexpr T log2e_v = std::numbers::log2e_v<T>;

        template<typename T>
        inline constexpr T log10e_v = std::numbers::log10e_v<T>;

        template<typename T>
        inline constexpr T pi_v = std::numbers::pi_v<T>;

        template<typename T>
        inline constexpr T inv_pi_v = std::numbers::inv_pi_v<T>;

        template<typename T>
        inline constexpr T ln2_v = std::numbers::ln2_v<T>;

        template<typename T>
        inline constexpr T ln10_v = std::numbers::ln10_v<T>;

        template<typename T>
        inline constexpr T sqrt2_v = std::numbers::sqrt2_v<T>;
#else
        inline constexpr double e = M_E;
        inline constexpr double log2e = M_LOG2E;
        inline constexpr double log10e = M_LOG10E;
        inline constexpr double pi = M_PI;
        inline constexpr double inv_pi = M_1_PI;
        inline constexpr double ln2 = M_LN2;
        inline constexpr double ln10 = M_LN10;
        inline constexpr double sqrt2 = M_SQRT2;

        template<typename T>
        inline constexpr T e_v = static_cast<T>(e);

        template<typename T>
        inline constexpr T log2e_v = static_cast<T>(log2e);

        template<typename T>
        inline constexpr T log10e_v = static_cast<T>(log10e);

        template<typename T>
        inline constexpr T pi_v = static_cast<T>(pi);

        template<typename T>
        inline constexpr T inv_pi_v = static_cast<T>(inv_pi);

        template<typename T>
        inline constexpr T ln2_v = static_cast<T>(ln2);

        template<typename T>
        inline constexpr T ln10_v = static_cast<T>(ln10);

        template<typename T>
        inline constexpr T sqrt2_v = static_cast<T>(sqrt2);

        // Use predefined float constants when available
#    if defined(M_Ef)
        template<>
        inline constexpr float e_v<float> = M_Ef;
#    endif

#    if defined(M_LOG2Ef)
        template<>
        inline constexpr float log2e_v<float> = M_LOG2Ef;
#    endif

#    if defined(M_LOG10Ef)
        template<>
        inline constexpr float log10e_v<float> = M_LOG10Ef;
#    endif

#    if defined(M_PIf)
        template<>
        inline constexpr float pi_v<float> = M_PIf;
#    endif

#    if defined(M_1_PIf)
        template<>
        inline constexpr float inv_pi_v<float> = M_1_PIf;
#    endif

#    if defined(M_LN2f)
        template<>
        inline constexpr float ln2_v<float> = M_LN2f;
#    endif

#    if defined(M_LN10f)
        template<>
        inline constexpr float ln10_v<float> = M_LN10f;
#    endif

#    if defined(M_SQRT2f)
        template<>
        inline constexpr float sqrt2_v<float> = M_SQRT2f;
#    endif

#endif
    } // namespace constants

} // namespace alpaka::math
