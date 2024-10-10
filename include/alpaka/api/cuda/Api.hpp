/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/HostApiConcepts.hpp"
#include "alpaka/core/config.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cuda
        {
            void _()
            {
                static_assert(concepts::Api<Cuda>);
            }

            static std::string getName()
            {
                return "Cuda";
            }
        };

        constexpr auto cuda = Cuda{};
    } // namespace api

#if ALPAKA_LANG_CUDA
    namespace trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::Cuda> : std::true_type
        {
        };
    } // namespace trait
#endif
} // namespace alpaka
