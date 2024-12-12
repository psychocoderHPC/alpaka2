/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/util.hpp"

#include <cassert>
#include <tuple>

namespace alpaka
{
    namespace object
    {
        struct Api
        {
        };

        constexpr Api api;

        ALPAKA_TAG(exec);
    } // namespace object

    namespace layer
    {
        ALPAKA_TAG(thread);
        ALPAKA_TAG(block);
        ALPAKA_TAG(shared);
    } // namespace layer

    namespace frame
    {
        ALPAKA_TAG(count);
        ALPAKA_TAG(extent);
    } // namespace frame

    namespace action
    {
        ALPAKA_TAG(sync);
    } // namespace action

    struct Empty
    {
    };

    namespace exec
    {


        namespace traits
        {
            template<typename T_Mapping>
            struct IsSeqExecutor : std::false_type
            {
            };

            template<typename T_Exec>
            constexpr bool isSeqExecutor_v = IsSeqExecutor<T_Exec>::value;

        } // namespace traits
    } // namespace exec

    /** check if a executor can only be used with a single thred per block
     *
     * @return true if a block can only have a single thread, else false
     */
    template<typename T_Exec>
    consteval bool isSeqExecutor(T_Exec exec)
    {
        return exec::traits::isSeqExecutor_v<T_Exec>;
    }
} // namespace alpaka
