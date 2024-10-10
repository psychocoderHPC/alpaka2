/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Layer.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka
{
    template<typename T_Acc, typename T_ExtentFn, typename T_StrideFn, typename T_StartIdxFn>
    class IndexContainer
    {
    public:
        using IdxVecType = decltype(T_ExtentFn{}(std::declval<T_Acc>()));
        using IdxType = typename IdxVecType::type;

        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc) : m_acc(acc), m_extent(T_ExtentFn{}(acc))
        {
        }

        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc, IdxVecType const& extent) : m_acc(acc), m_extent(extent)
        {
        }

        class const_iterator;
        using iterator = const_iterator;

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            return const_iterator(T_StrideFn{}(m_acc).x(), m_extent.x(), T_StartIdxFn{}(m_acc).x());
        }

        ALPAKA_FN_ACC inline const_iterator end() const
        {
            return const_iterator(T_StrideFn{}(m_acc).x(), m_extent.x(), m_extent.x());
        }

        class const_iterator
        {
            friend class IndexContainer;

            ALPAKA_FN_ACC inline const_iterator(IdxType stride, IdxType extent, IdxType first)
                : m_stride{stride}
                , m_extent{extent}
                , m_current{std::min(first, extent)}
            {
            }

        public:
            ALPAKA_FN_ACC inline IdxType operator*() const
            {
                return m_current;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                // increment the first-element-in-block index by the grid stride
                m_current += m_stride;
                if(m_current > m_extent)
                    m_current = m_extent;
                return *this;
            }

            // post-increment the iterator
            ALPAKA_FN_ACC inline const_iterator operator++(int)
            {
                const_iterator old = *this;
                ++(*this);
                return old;
            }

            ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const
            {
                return (m_current == other.m_current);
            }

            ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const
            {
                return not(*this == other);
            }

        private:
            // non-const to support iterator copy and assignment
            IdxType m_stride;
            IdxType m_extent;
            // modified by the pre/post-increment operator
            IdxType m_current;
        };

    private:
        T_Acc const& m_acc;
        IdxVecType const m_extent;
    };

    namespace idxTrait
    {
        struct DataExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[frame::block] * acc[frame::thread];
            }
        };

        struct GlobalThreadIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
            }
        };

        struct GlobalNumThreads
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::block].count() * acc[layer::thread].count();
            }
        };
    } // namespace idxTrait

    template<typename T_Acc>
    using IndependentDataIter
        = IndexContainer<T_Acc, idxTrait::DataExtent, idxTrait::GlobalNumThreads, idxTrait::GlobalThreadIdx>;

    template<typename T_Acc>
    using IndependentGridThreadIter
        = IndexContainer<T_Acc, idxTrait::GlobalNumThreads, idxTrait::GlobalNumThreads, idxTrait::GlobalThreadIdx>;

} // namespace alpaka
