/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"

#include <cassert>
#include <experimental/mdspan>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        template<typename ElementType>
        struct ByteIndexedAccessor
        {
            using offset_policy = ByteIndexedAccessor;
            using element_type = ElementType;
            using reference = ElementType&;

            using data_handle_type = std::conditional_t<std::is_const_v<ElementType>, std::byte const*, std::byte*>;

            constexpr ByteIndexedAccessor() noexcept = default;

            constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept
            {
                return p + i;
            }

            constexpr reference access(data_handle_type p, size_t i) const noexcept
            {
                assert(i % alignof(ElementType) == 0);
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wcast-align"
#endif
                return *reinterpret_cast<ElementType*>(p + i);
#if ALPAKA_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
            }
        };

        template<typename T_Extnets, std::size_t... Is>
        constexpr auto makeExtents(T_Extnets const& extent, std::index_sequence<Is...>)
        {
            auto const ex = extent;
            return std::experimental::dextents<typename T_Extnets::type, T_Extnets::dim()>{ex[Is]...};
        }

    } // namespace detail

    template<typename T>
    struct MdSpan : T
    {
        constexpr MdSpan(T const& base) : T{base}
        {
        }

        template<typename T_Type, uint32_t T_dim>
        constexpr decltype(auto) operator[](Vec<T_Type, T_dim> const& vec) const
        {
            return T::operator()(vec.toStdArray());
        }

        template<typename T_Type>
        requires(std::is_integral_v<T_Type> && T::rank() == 1u)
        constexpr decltype(auto) operator[](T_Type const& value) const
        {
            return T::operator()(value);
        }
    };
} // namespace alpaka
