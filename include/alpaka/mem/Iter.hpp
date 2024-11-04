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

        static constexpr uint32_t dim = IdxVecType::dim();

        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc) : m_acc(acc), m_extent(T_ExtentFn{}(acc))
        {
        }

        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc, IdxVecType const& extent) : m_acc(acc), m_extent(extent)
        {
        }

        class const_iterator;

        /** special implementation to define the end
         *
         * Only a scalar value must be stored which reduce the register footprint.
         * The definition of end is that the index is behind or equal to the extent of the slowest moving dimension.
         */
        class const_iterator_end
        {
            friend class IndexContainer;
            friend class const_iterator;

            ALPAKA_FN_ACC inline const_iterator_end(IdxType extentSlowDim) : m_extentSlowDim{extentSlowDim}
            {
            }

        public:
            ALPAKA_FN_ACC inline bool operator==(const_iterator_end const& other) const
            {
                return (m_extentSlowDim == other.m_extentSlowDim);
            }

            ALPAKA_FN_ACC inline bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

            ALPAKA_FN_ACC inline bool operator==(const_iterator const& other) const
            {
                return (m_extentSlowDim <= other.m_extent[0]);
            }

            ALPAKA_FN_ACC inline bool operator!=(const_iterator const& other) const
            {
                return not(*this == other);
            }

        private:
            IdxType m_extentSlowDim;
        };

        class const_iterator
        {
            friend class IndexContainer;
            friend class const_iterator_end;

            ALPAKA_FN_ACC inline const_iterator(IdxVecType stride, IdxVecType extent, IdxVecType first)
                : m_stride{stride}
                , m_first{first}
                , m_extent{extent}
                , m_current{first}
            {
                // invalidate current if one dimension is out of range.
                bool isIndexValid = true;
                for(uint32_t d = 1u; d < dim; ++d)
                    isIndexValid = isIndexValid && (first[d] < extent[d]);
                if(!isIndexValid)
                    m_current[0] = m_extent[0];
            }

        public:
            ALPAKA_FN_ACC inline IdxVecType operator*() const
            {
                return m_current;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                for(uint32_t d = 0; d < dim; ++d)
                {
                    uint32_t const idx = dim - 1u - d;
                    m_current[idx] += m_stride[idx];
                    if(m_current[idx] >= m_extent[idx])
                    {
                        if(idx >= 1u)
                            m_current[idx] = m_first[idx];
                    }
                    else
                        break;
                }
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

            ALPAKA_FN_ACC inline bool operator==(const_iterator_end const& other) const
            {
                return (m_current[0] >= other.m_extentSlowDim);
            }

            ALPAKA_FN_ACC inline bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

        private:
            // non-const to support iterator copy and assignment
            IdxVecType m_stride;
            IdxVecType m_first;
            IdxVecType m_extent;
            // modified by the pre/post-increment operator
            IdxVecType m_current;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            return const_iterator(T_StrideFn{}(m_acc), m_extent, T_StartIdxFn{}(m_acc));
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            return const_iterator_end(m_extent[0]);
        }

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
                return acc[frame::count] * acc[frame::extent];
            }
        };

        struct DataFrameCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[frame::count];
            }
        };

        struct DataFrameExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[frame::extent];
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

        struct GlobalThreadBlockIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::block].idx();
            }
        };

        struct GlobalNumThreadBlocks
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::block].count();
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

        struct NumThreadsInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::thread].count();
            }
        };

        struct ThreadIdxInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc)
            {
                return acc[layer::thread].idx();
            }
        };
    } // namespace idxTrait

    template<typename T_Acc>
    using IndependentDataIter
        = IndexContainer<T_Acc, idxTrait::DataExtent, idxTrait::GlobalNumThreads, idxTrait::GlobalThreadIdx>;

    template<typename T_Acc>
    using DataBlockIter = IndexContainer<
        T_Acc,
        idxTrait::DataFrameCount,
        idxTrait::GlobalNumThreadBlocks,
        idxTrait::GlobalThreadBlockIdx>;

    template<typename T_Acc>
    using DataFrameIter
        = IndexContainer<T_Acc, idxTrait::DataFrameExtent, idxTrait::NumThreadsInBlock, idxTrait::ThreadIdxInBlock>;

    template<typename T_Acc>
    using IndependentGridThreadIter
        = IndexContainer<T_Acc, idxTrait::GlobalNumThreads, idxTrait::GlobalNumThreads, idxTrait::GlobalThreadIdx>;

} // namespace alpaka
