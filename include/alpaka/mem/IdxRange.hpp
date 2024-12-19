/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka
{

    template<
        concepts::Vector T_End,
        concepts::Vector T_Begin = typename T_End::UniVec,
        concepts::Vector T_Stride = typename T_End::UniVec>
    struct IdxRange
    {
        using IdxType = typename T_End::type;
        using IdxVecType = typename T_End::UniVec;

        constexpr IdxRange(T_Begin const& begin, T_End const& end, T_Stride const& stride)
            : m_begin{begin}
            , m_end{end}
            , m_stride{stride}
        {
        }

        constexpr IdxRange(T_Begin const& begin, T_End const& end)
            : m_begin{begin}
            , m_end{end}
            , m_stride{T_End::all(1u)}
        {
        }

        constexpr IdxRange(T_End const& extent) : m_begin{T_End::all(0u)}, m_end{extent}, m_stride{T_End::all(1u)}
        {
        }

        static consteval uint32_t dim()
        {
            return IdxVecType::dim();
        }

        template<concepts::TypeOrVector<typename T_End::type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator%(T_OpType const& rhs) const
        {
            return IdxRange<T_End, T_Begin, ALPAKA_TYPEOF(m_stride * rhs)>{m_begin, m_end, m_stride * rhs};
        }

        template<concepts::TypeOrVector<typename T_End::type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator>>(T_OpType const& rhs) const
        {
            return IdxRange<ALPAKA_TYPEOF(m_end + rhs), ALPAKA_TYPEOF(m_begin + rhs), ALPAKA_TYPEOF(m_stride)>{
                m_begin + rhs,
                m_end + rhs,
                m_stride};
        }

        template<concepts::TypeOrVector<typename T_End::type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr auto operator<<(T_OpType const& rhs) const
        {
            return IdxRange<ALPAKA_TYPEOF(m_end - rhs), ALPAKA_TYPEOF(m_begin - rhs), T_Stride>{
                m_begin - rhs,
                m_end - rhs,
                m_stride};
        }

        constexpr auto distance() const
        {
            return m_end - m_begin;
        }

        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string locale_enclosing_begin;
            std::string locale_enclosing_end;
            size_t enclosing_dim = enclosings.size();

            if(enclosing_dim > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosing_dim];
                locale_enclosing_end = enclosings[1 % enclosing_dim];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin;
            stream << m_begin << separator << m_end << separator << m_stride;
            stream << locale_enclosing_end;
            return stream.str();
        }

        T_Begin m_begin;
        T_End m_end;
        T_Stride m_stride;
    };

    template<concepts::VectorOrScalar T_Extent>
    IdxRange(T_Extent const&) -> IdxRange<typename trait::getVec_t<T_Extent>::UniVec>;

    template<concepts::VectorOrScalar T_Begin, concepts::VectorOrScalar T_End>
    IdxRange(T_Begin const&, T_End const&)
        -> IdxRange<typename trait::getVec_t<T_Begin>::UniVec, typename trait::getVec_t<T_End>::UniVec>;

    template<concepts::VectorOrScalar T_Begin, concepts::VectorOrScalar T_End, concepts::VectorOrScalar T_Stride>
    IdxRange(T_Begin const&, T_End const&, T_Stride const&) -> IdxRange<
        typename trait::getVec_t<T_Begin>::UniVec,
        typename trait::getVec_t<T_End>::UniVec,
        typename trait::getVec_t<T_Stride>::UniVec>;

} // namespace alpaka
