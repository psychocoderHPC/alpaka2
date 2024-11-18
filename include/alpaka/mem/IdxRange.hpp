/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka
{

    template<typename T_Begin, typename T_End, typename T_Stride>
    struct IdxRange
    {
        using IdxVecType = T_Begin;
        using IdxType = typename IdxVecType::type;

        constexpr IdxRange(T_Begin const& begin, T_End const& end, T_Stride const& stride)
            : m_begin(begin)
            , m_end(end)
            , m_stride(stride)
        {
        }

        static consteval uint32_t dim()
        {
            return T_Begin::dim();
        }

        /** assign operator
         * @{
         */
#define ALPAKA_ITER_ASSIGN_OP(interfaceOp, executedOp)                                                                \
    template<concepts::TypeOrVector<typename T_Begin::type> T_OpType>                                                 \
    ALPAKA_FN_HOST_ACC constexpr IdxRange& operator interfaceOp(T_OpType const& rhs)                                  \
    {                                                                                                                 \
        m_begin executedOp rhs;                                                                                       \
        m_end executedOp rhs;                                                                                         \
        return *this;                                                                                                 \
    }

        ALPAKA_ITER_ASSIGN_OP(>>=, +=)
        ALPAKA_ITER_ASSIGN_OP(<<=, -=)
#undef ALPAKA_ITER_ASSIGN_OP

        template<concepts::TypeOrVector<typename T_Stride::type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr IdxRange& operator%=(T_OpType const& rhs)
        {
            m_stride *= rhs;
            return *this;
        }

        template<concepts::TypeOrVector<typename T_Stride::type> T_OpType>
        ALPAKA_FN_HOST_ACC constexpr IdxRange operator%(T_OpType const& rhs)
        {
            auto idxContainer = (*this);
            idxContainer.m_stride *= rhs;
            return idxContainer;
        }

#define ALPAKA_ITER_BINARY_OP(op)                                                                                     \
    template<concepts::TypeOrVector<typename T_Begin::type> T_OpType>                                                 \
    ALPAKA_FN_HOST_ACC constexpr IdxRange operator op(T_OpType const& rhs)                                            \
    {                                                                                                                 \
        auto idxContainer = (*this);                                                                                  \
        idxContainer ALPAKA_PP_CAT(op, =) rhs;                                                                        \
        return idxContainer;                                                                                          \
    }

        ALPAKA_ITER_BINARY_OP(>>)
        ALPAKA_ITER_BINARY_OP(<<)

#undef ALPAKA_ITER_BINARY_OP

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
} // namespace alpaka
