/* Copyright 2020 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"

#include <algorithm>
#include <type_traits>

namespace alpaka::onAcc
{
    //! The addition function object.
    struct AtomicAdd
    {
        //! \return The old value of addr.
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#endif
            ref += value;
            return old;
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
        }
    };

    //! The subtraction function object.
    struct AtomicSub
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#endif
            ref -= value;
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
            return old;
        }
    };

    //! The minimum function object.
    struct AtomicMin
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref = std::min(ref, value);
            return old;
        }
    };

    //! The maximum function object.
    struct AtomicMax
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref = std::max(ref, value);
            return old;
        }
    };

    //! The exchange function object.
    struct AtomicExch
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref = value;
            return old;
        }
    };

    //! The increment function object.
    struct AtomicInc
    {
        //! Increments up to value, then reset to 0.
        //!
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref = ((old >= value) ? static_cast<T>(0) : static_cast<T>(old + static_cast<T>(1)));
            return old;
        }
    };

    //! The decrement function object.
    struct AtomicDec
    {
        //! Decrement down to 0, then reset to value.
        //!
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref = (((old == static_cast<T>(0)) || (old > value)) ? value : static_cast<T>(old - static_cast<T>(1)));
            return old;
        }
    };

    //! The and function object.
    struct AtomicAnd
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref &= value;
            return old;
        }
    };

    //! The or function object.
    struct AtomicOr
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref |= value;
            return old;
        }
    };

    //! The exclusive or function object.
    struct AtomicXor
    {
        //! \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T>
        ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;
            ref ^= value;
            return old;
        }
    };

    //! The compare and swap function object.
    struct AtomicCas
    {
        //! AtomicCas for non floating point values
        // \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, std::enable_if_t<!std::is_floating_point_v<T>, bool> = true>
        ALPAKA_FN_HOST_ACC auto operator()(T* addr, T const& compare, T const& value) const -> T
        {
            auto const old = *addr;
            auto& ref = *addr;

            // check if values are bit-wise equal
            ref = ((old == compare) ? value : old);
            return old;
        }

        //! AtomicCas for floating point values
        // \return The old value of addr.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
        ALPAKA_FN_HOST_ACC auto operator()(T* addr, T const& compare, T const& value) const -> T
        {
            static_assert(sizeof(T) == 4u || sizeof(T) == 8u, "AtomicCas is supporting only 32bit and 64bit values!");
            // Type to reinterpret too to perform the bit comparison
            using BitType = std::conditional_t<sizeof(T) == 4u, unsigned int, unsigned long long>;

            // type used to have a safe way to reinterprete the data into another type
            // std::variant can not be used because clang8 has issues to compile std::variant
            struct BitUnion
            {
                union
                {
                    T value;
                    BitType r;
                };
            };

            auto const old = *addr;
            auto& ref = *addr;


            BitUnion o{old};
            BitUnion c{compare};

            ref = ((o.r == c.r) ? value : old);
            return old;
        }
    };
} // namespace alpaka::onAcc
