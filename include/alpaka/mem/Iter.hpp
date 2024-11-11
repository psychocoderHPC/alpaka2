/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Tags.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka
{
    namespace iter::detail
    {
        /** Store reduced vector
         *
         * The first index can be reduced by on dimension because the slowest dimension must never set to zero after
         * the initialization.
         */
        template<typename T_Type, uint32_t T_dim>
        struct ReducedVector : private Vec<T_Type, T_dim - 1u>
        {
            constexpr ReducedVector(Vec<T_Type, T_dim> const& first)
                : Vec<T_Type, T_dim - 1u>{first.template rshrink<T_dim - 1u>()}
            {
            }

            constexpr decltype(auto) operator[](T_Type idx) const
            {
                return Vec<T_Type, T_dim - 1u>::operator[](idx - 1u);
            }

            constexpr decltype(auto) operator[](T_Type idx)
            {
                return Vec<T_Type, T_dim - 1u>::operator[](idx - 1u);
            }
        };

        template<typename T_Type>
        struct ReducedVector<T_Type, 1u>
        {
            constexpr ReducedVector(Vec<T_Type, 1u> const&)
            {
            }
        };
    } // namespace iter::detail

    struct Stride
    {
        auto adjust(auto const& extentVec, auto const& strideVec) const
        {
            return std::make_tuple(extentVec, strideVec);
        }
    };

    template<typename T_MapperFn, typename IdxVecType, typename T_StartIdxFn, typename T_ExtentFn, typename T_StrideFn>
    class IndexContainer
    {
        void _()
        {
            static_assert(std::ranges::forward_range<IndexContainer>);
        }

    public:
        using IdxType = typename IdxVecType::type;

        static constexpr uint32_t dim = IdxVecType::dim();

        template<typename T_Acc>
        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc)
            : m_extent(T_ExtentFn{}(acc))
            , m_stride{T_StrideFn{}(acc)}
            , m_first{T_StartIdxFn{}(acc)}
        {
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc, IdxVecType const& extent)
            : m_extent(extent)
            , m_stride{T_StrideFn{}(acc)}
            , m_first{T_StartIdxFn{}(acc)}
        {
        }

        template<typename T_Acc>
        ALPAKA_FN_ACC inline IndexContainer(T_Acc const& acc, IdxVecType const& extent, IdxVecType const& offset)
            : m_extent(extent)
            , m_stride{T_StrideFn{}(acc)}
            , m_first{T_StartIdxFn{}(acc)}
            , m_offset(offset)
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

            void _()
            {
                static_assert(std::forward_iterator<const_iterator_end>);
            }

            ALPAKA_FN_ACC inline const_iterator_end(IdxType extentSlowDim) : m_extentSlowDim{extentSlowDim}
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
                return (m_extentSlowDim <= other.m_extent[0]);
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
            friend class IndexContainer;
            friend class const_iterator_end;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator>);
            }

            constexpr const_iterator(IdxVecType stride, IdxVecType extent, IdxVecType first)
                : m_stride{stride}
                , m_extent{extent}
                , m_current{first}
                , m_first(first)
            {
                // range check required for 1 dimensional iterators
                if constexpr(dim > 1u)
                {
                    // invalidate current if one dimension is out of range.
                    bool isIndexValid = true;
                    for(uint32_t d = 1u; d < dim; ++d)
                        isIndexValid = isIndexValid && (first[d] < extent[d]);
                    if(!isIndexValid)
                        m_current[0] = m_extent[0];
                }
            }

        public:
            constexpr IdxVecType operator*() const
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
                    if constexpr(dim != 1u)
                    {
                        if(idx >= 1u && m_current[idx] >= m_extent[idx])
                        {
                            m_current[idx] = m_first[idx];
                        }
                        else
                            break;
                    }
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

            constexpr bool operator==(const_iterator const& other) const
            {
                return (m_current == other.m_current);
            }

            constexpr bool operator!=(const_iterator const& other) const
            {
                return not(*this == other);
            }

            constexpr bool operator==(const_iterator_end const& other) const
            {
                return (m_current[0] >= *other);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

        private:
            // non-const to support iterator copy and assignment
            IdxVecType m_stride;
            IdxVecType m_extent;
            // modified by the pre/post-increment operator
            IdxVecType m_current;
            iter::detail::ReducedVector<IdxType, dim> m_first;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            return const_iterator(m_stride, m_extent + m_offset, m_first + m_offset);
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            return const_iterator_end(m_extent[0] + m_offset[0]);
        }

    private:
        IdxVecType m_extent;
        IdxVecType m_stride;
        IdxVecType m_first;
        IdxVecType m_offset;
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

    struct Auto
    {
    };

    struct MakeIterator
    {
        template<typename T_IdxMapping, typename T_StartIdxFn, typename T_ExtentFn, typename T_StrideFn>
        struct Create
        {
            template<typename T_Acc>
            ALPAKA_FN_ACC static auto get(T_Acc const& acc)
            {
                using IdxVecType = decltype(T_ExtentFn{}(std::declval<T_Acc>()));
                if constexpr(std::is_same_v<T_IdxMapping, Auto>)
                {
                    if constexpr(std::is_same_v<api::Cpu, decltype(acc[object::api])>)
                        return IndexContainer<Stride, IdxVecType, T_StartIdxFn, T_ExtentFn, T_StrideFn>{acc};
                    else
                        return IndexContainer<Stride, IdxVecType, T_StartIdxFn, T_ExtentFn, T_StrideFn>{acc};
                }
            }

            template<typename T_Acc, typename T_IdxVec>
            ALPAKA_FN_ACC static auto get(T_Acc const& acc, T_IdxVec const& extent)
            {
                if constexpr(std::is_same_v<T_IdxMapping, Auto>)
                {
                    if constexpr(std::is_same_v<api::Cpu, decltype(acc[object::api])>)
                        return IndexContainer<Stride, T_IdxVec, T_StartIdxFn, T_ExtentFn, T_StrideFn>{acc, extent};
                    else
                        return IndexContainer<Stride, T_IdxVec, T_StartIdxFn, T_ExtentFn, T_StrideFn>{acc, extent};
                }
            }

            template<typename T_Acc, typename T_IdxVec>
            ALPAKA_FN_ACC static auto get(T_Acc const& acc, T_IdxVec const& offset, T_IdxVec const& extent)
            {
                if constexpr(std::is_same_v<T_IdxMapping, Auto>)
                {
                    if constexpr(std::is_same_v<api::Cpu, decltype(acc[object::api])>)
                        return IndexContainer<Stride, T_IdxVec, T_StartIdxFn, T_ExtentFn, T_StrideFn>{
                            acc,
                            extent,
                            offset};
                    else
                        return IndexContainer<Stride, T_IdxVec, T_StartIdxFn, T_ExtentFn, T_StrideFn>{
                            acc,
                            extent,
                            offset};
                }
            }
        };
    };

    template<typename T_IdxMapping = Auto>
    using IndependentDataIter = MakeIterator::
        Create<T_IdxMapping, idxTrait::GlobalThreadIdx, idxTrait::DataExtent, idxTrait::GlobalNumThreads>;

    template<typename T_IdxMapping = Auto>
    using DataBlockIter = MakeIterator::
        Create<T_IdxMapping, idxTrait::GlobalThreadBlockIdx, idxTrait::DataFrameCount, idxTrait::GlobalNumThreadBlocks>;

    template<typename T_IdxMapping = Auto>
    using DataFrameIter = MakeIterator::
        Create<T_IdxMapping, idxTrait::ThreadIdxInBlock, idxTrait::DataFrameExtent, idxTrait::NumThreadsInBlock>;

    template<typename T_IdxMapping = Auto>
    using IndependentGridThreadIter = MakeIterator::
        Create<T_IdxMapping, idxTrait::GlobalThreadIdx, idxTrait::GlobalNumThreads, idxTrait::GlobalNumThreads>;

    template<typename T_IdxMapping = Auto>
    using IndependentBlockThreadIter = MakeIterator::
        Create<T_IdxMapping, idxTrait::ThreadIdxInBlock, idxTrait::NumThreadsInBlock, idxTrait::NumThreadsInBlock>;

} // namespace alpaka
