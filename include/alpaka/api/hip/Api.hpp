/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/unifiedCudaHip/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Hip
        {
            using element_type = Hip;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Hip>);
            }

            static std::string getName()
            {
                return "Hip";
            }
        };

        constexpr auto hip = Hip{};
    } // namespace api

    namespace onHost::trait
    {
#if ALPAKA_LANG_HIP
        template<>
        struct IsPlatformAvailable::Op<api::Hip> : std::true_type
        {
        };
#endif
    } // namespace onHost::trait

    namespace unifiedCudaHip::trait
    {
        template<>
        struct IsUnifiedApi<api::Hip> : std::true_type
        {
        };
    } // namespace unifiedCudaHip::trait
} // namespace alpaka
