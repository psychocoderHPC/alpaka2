/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/HostApiConcepts.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cpu
        {
            void _()
            {
                static_assert(concepts::Api<Cpu>);
            }

            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};
    } // namespace api

    namespace trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::Cpu> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka
