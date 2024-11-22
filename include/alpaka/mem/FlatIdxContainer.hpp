/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka::onAcc::iter
{

    template<typename T_IdxRange, typename T_ThreadSpace, typename T_IdxMapperFn, concepts::CVector T_CSelect>
    class FlatIdxContainer : private T_IdxMapperFn
    {
        void _()
        {
            static_assert(std::ranges::forward_range<FlatIdxContainer>);
        }

    public:
        using IdxType = typename T_IdxRange::IdxType;
        static constexpr uint32_t dim = T_IdxRange::dim();
        using IdxVecType = Vec<IdxType, dim>;

        ALPAKA_FN_ACC inline FlatIdxContainer(
            T_IdxRange const& idxRange,
            T_ThreadSpace const& threadSpace,
            T_IdxMapperFn idxMapping,
            T_CSelect const& = T_CSelect{})
            : T_IdxMapperFn{std::move(idxMapping)}
            , m_idxRange(idxRange)
            , m_threadSpace{threadSpace}
        {
            //  std::cout << "iter:" << m_idxRange.toString() << " " << m_threadSpace.toString() << std::endl;
        }

        class const_iterator;

        /** special implementation to define the end
         *
         * Only a scalar value must be stored which reduce the register footprint.
         * The definition of end is that the index is behind or equal to the extent of the slowest moving dimension.
         */
        class const_iterator_end
        {
            friend class FlatIdxContainer;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator_end>);
            }

            ALPAKA_FN_ACC inline const_iterator_end(concepts::Vector auto const& extent)
                : m_extentSlowDim{extent.select(T_CSelect{})[0]}
            {
            }

            constexpr IdxType operator*() const
            {
                return m_extentSlowDim;
            }

        public:
            constexpr bool operator==(const_iterator_end const& other) const
            {
                return (m_extentSlowDim == other.m_extentSlowDim);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

            constexpr bool operator==(const_iterator const& other) const
            {
                return (m_extentSlowDim <= other.slowCurrent);
            }

            constexpr bool operator!=(const_iterator const& other) const
            {
                return not(*this == other);
            }

        private:
            IdxType m_extentSlowDim;
        };

        class const_iterator
        {
            friend class FlatIdxContainer;
            friend class const_iterator_end;

            static constexpr uint32_t iterDim = T_CSelect::dim();
            using IterIdxVecType = Vec<IdxType, iterDim>;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator>);
            }

            constexpr const_iterator(
                concepts::Vector auto const offset,
                concepts::Vector auto const first,
                concepts::Vector auto const extent,
                concepts::Vector auto const stride)
                : m_current{linearize(stride.select(T_CSelect{}), first.select(T_CSelect{}))}
                , m_stride{stride.select(T_CSelect{}).product()}
                , m_extent{extent.select(T_CSelect{})}
                , m_offset(first + offset)
            {
                m_offset.ref(T_CSelect{}) -= first.select(T_CSelect{});
            }

            ALPAKA_FN_ACC constexpr IdxType slowCurrent() const
            {
                return (**this)[T_CSelect{}[0]];
            }

        public:
            constexpr IdxVecType operator*() const
            {
                auto result = m_offset;
                result.ref(T_CSelect{}) += mapToND(m_extent, m_current);
                return result;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                m_current += m_stride;
                return *this;
            }

            // post-increment the iterator
            ALPAKA_FN_ACC inline const_iterator operator++(int)
            {
                const_iterator old = *this;
                ++(*this);
                return old;
            }

            constexpr bool operator==(const_iterator const& other) const
            {
                return ((**this) == *other);
            }

            constexpr bool operator!=(const_iterator const& other) const
            {
                return not(*this == other);
            }

            constexpr bool operator==(const_iterator_end const& other) const
            {
                return (slowCurrent() >= *other);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

        private:
            // modified by the pre/post-increment operator
            IdxType m_current;
            // non-const to support iterator copy and assignment
            IdxType m_stride;
            IterIdxVecType m_extent;
            IdxVecType m_offset;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            auto [first, extent, stride] = this->adjust(
                m_idxRange.m_stride,
                m_idxRange.distance(),
                m_threadSpace.m_threadIdx,
                m_threadSpace.m_threadCount);
            return const_iterator(m_idxRange.m_begin, first, extent, stride);
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            auto [_, extent, __] = this->adjust(
                m_idxRange.m_stride,
                m_idxRange.distance(),
                m_threadSpace.m_threadIdx,
                m_threadSpace.m_threadCount);
            return const_iterator_end(m_idxRange.m_begin + extent);
        }

        ALPAKA_FN_HOST_ACC constexpr auto operator[](concepts::CVector auto const iterDir) const
        {
            return FlatIdxContainer<T_IdxRange, T_ThreadSpace, T_IdxMapperFn, ALPAKA_TYPE(iterDir)>(
                m_idxRange,
                m_threadSpace,
                T_IdxMapperFn{});
        }

    private:
        T_IdxRange m_idxRange;
        T_ThreadSpace m_threadSpace;
    };
} // namespace alpaka::onAcc::iter
