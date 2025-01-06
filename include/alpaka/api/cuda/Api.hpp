/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/concepts.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cuda
        {
            using element_type = Cuda;

            auto get() const
            {
                return this;
            }

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
#if ALPAKA_LANG_CUDA
        template<>
        struct IsPlatformAvailable::Op<api::Cuda> : std::true_type
        {
        };
#endif
#if ALPAKA_LANG_HIP
        template<>
        struct IsPlatformAvailable::Op<api::Hip> : std::true_type
        {
        };
#endif
    } // namespace onHost::trait
} // namespace alpaka
