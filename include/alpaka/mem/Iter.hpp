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
        static constexpr auto adjust(auto const& firstVec, auto const& extentVec, auto const& strideVec)
        {
            // std::cout<<firstVec<<extentVec<<strideVec<<std::endl;
            return std::make_tuple(firstVec, extentVec, strideVec);
        }
    };

    struct Contigious
    {
        static constexpr auto adjust(auto const& firstVec, auto const& extentVec, auto const& strideVec)
        {
            auto numElements = core::divCeil(extentVec, strideVec);
            auto stride = ALPAKA_TYPE(strideVec)::all(1u);
            auto first = firstVec * numElements;
            // std::cout<<firstVec<<first<<first + numElements<<stride<<std::endl;
            return std::make_tuple(first, extentVec.min(first + numElements), stride);
        }
    };

    struct Optimize
    {
    };

    constexpr auto optimize = Optimize{};

    template<typename T>
    struct IsIdxMapping : std::false_type
    {
    };

    template<>
    struct IsIdxMapping<Stride> : std::true_type
    {
    };

    template<>
    struct IsIdxMapping<Optimize> : std::true_type
    {
    };

    template<>
    struct IsIdxMapping<Contigious> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isIdxMapping_v = IsIdxMapping<T>::value;

    namespace concepts
    {
        template<typename T>
        concept IdxMapping = isIdxMapping_v<T>;
    } // namespace concepts

    template<typename T_IdxVecType, typename T_IdxMapperFn>
    class IndexContainer : private T_IdxMapperFn
    {
        void _()
        {
            static_assert(std::ranges::forward_range<IndexContainer>);
        }

    public:
        using IdxVecType = T_IdxVecType;
        using IdxType = typename IdxVecType::type;

        static constexpr uint32_t dim = IdxVecType::dim();

        ALPAKA_FN_ACC inline IndexContainer(
            IdxVecType const& first,
            IdxVecType const& extent,
            IdxVecType const& stride,
            IdxVecType const& offset,
            T_IdxMapperFn idxMapping)
            : T_IdxMapperFn{std::move(idxMapping)}
            , m_extent(extent)
            , m_stride{stride}
            , m_first{first}
            , m_offset{offset}
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
            auto [first, extent, stride] = this->adjust(m_first, m_extent, m_stride);
            return const_iterator(stride, extent + m_offset, first + m_offset);
        }

        ALPAKA_FN_ACC inline const_iterator_end end() const
        {
            auto [_, extent, __] = this->adjust(m_first, m_extent, m_stride);
            return const_iterator_end(extent[0] + m_offset[0]);
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

    ALPAKA_TAG(firstFn);
    ALPAKA_TAG(extentFn);
    ALPAKA_TAG(strideFn);

    namespace internal
    {
        struct AutoIndexMapping
        {
            template<typename T_Acc, typename T_Api>
            struct Op
            {
                constexpr auto operator()(T_Acc const&, T_Api) const
                {
                    return Stride{};
                }
            };
        };

        template<typename T_Acc>
        requires(not exec::traits::isSeqMapping_v<ALPAKA_TYPE(std::declval<T_Acc>()[object::exec])>)
        struct AutoIndexMapping::Op<T_Acc, api::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Cpu) const
            {
                return Contigious{};
            }
        };

        constexpr auto adjustMapping(auto const& acc, auto api)
        {
            return internal::AutoIndexMapping::Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(api)>{}(acc, api);
        }

        struct MakeIter
        {
            template<typename T_Acc, typename T_RangeOps, typename T_IdxMapping>
            struct Op
            {
                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping) const
                {
                    return (*this)(acc, rangeOps, idxMapping, rangeOps[extentFn](acc));
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping,
                    concepts::Vector auto const& extent) const
                {
                    return (*this)(acc, rangeOps, idxMapping, ALPAKA_TYPE(extent)::all(0), extent);
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping,
                    concepts::Vector auto const& offset,
                    concepts::Vector auto const& extent) const
                    requires std::is_same_v<ALPAKA_TYPE(idxMapping), ALPAKA_TYPE(optimize)>
                {
                    static_assert(std::is_same_v<ALPAKA_TYPE(offset), ALPAKA_TYPE(extent)>);

                    auto adjIdxMapping = adjustMapping(acc, idxMapping);
                    return IndexContainer<ALPAKA_TYPE(extent), ALPAKA_TYPE(adjIdxMapping)>{
                        rangeOps[firstFn](acc),
                        extent,
                        rangeOps[strideFn](acc),
                        offset,
                        adjIdxMapping};
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping,
                    concepts::Vector auto const& offset,
                    concepts::Vector auto const& extent) const
                {
                    static_assert(std::is_same_v<ALPAKA_TYPE(offset), ALPAKA_TYPE(extent)>);

                    return IndexContainer<ALPAKA_TYPE(extent), ALPAKA_TYPE(idxMapping)>(
                        rangeOps[firstFn](acc),
                        extent,
                        rangeOps[strideFn](acc),
                        offset,
                        idxMapping);
                }
            };
        };
    } // namespace internal

    template<concepts::IdxMapping T_IdxMapping = Optimize>
    ALPAKA_FN_HOST_ACC constexpr auto makeIter(
        auto const& acc,
        auto rangeOps,
        T_IdxMapping idxMapping = T_IdxMapping{})
    {
        return internal::MakeIter::Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(rangeOps), T_IdxMapping>{}(
            acc,
            rangeOps,
            idxMapping);
    }

    template<concepts::IdxMapping T_IdxMapping = Optimize>
    ALPAKA_FN_HOST_ACC constexpr auto makeIter(
        auto const& acc,
        auto rangeOps,
        concepts::Vector auto const& extent,
        T_IdxMapping idxMapping = T_IdxMapping{})
    {
        return internal::MakeIter::Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(rangeOps), T_IdxMapping>{}(
            acc,
            rangeOps,
            idxMapping,
            extent);
    }

    template<concepts::IdxMapping T_IdxMapping = Optimize>
    ALPAKA_FN_HOST_ACC constexpr auto makeIter(
        auto const& acc,
        auto rangeOps,
        concepts::Vector auto const& offset,
        concepts::Vector auto const& extent,
        T_IdxMapping idxMapping = T_IdxMapping{})
    {
        return internal::MakeIter::Op<ALPAKA_TYPE(acc), ALPAKA_TYPE(rangeOps), T_IdxMapping>{}(
            acc,
            rangeOps,
            idxMapping,
            offset,
            extent);
    }

    namespace iter
    {
        constexpr auto overDataRange = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadIdx{}),
            DictEntry(extentFn, idxTrait::DataExtent{}),
            DictEntry(strideFn, idxTrait::GlobalNumThreads{}))};

        constexpr auto overDataFrames = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadBlockIdx{}),
            DictEntry(extentFn, idxTrait::DataFrameCount{}),
            DictEntry(strideFn, idxTrait::GlobalNumThreadBlocks{}))};

        constexpr auto withinDataFrame = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::ThreadIdxInBlock{}),
            DictEntry(extentFn, idxTrait::DataFrameExtent{}),
            DictEntry(strideFn, idxTrait::NumThreadsInBlock{}))};

        constexpr auto overThreadRange = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadIdx{}),
            DictEntry(extentFn, idxTrait::GlobalNumThreads{}),
            DictEntry(strideFn, idxTrait::GlobalNumThreads{}))};

        constexpr auto overThreadBlocks = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadBlockIdx{}),
            DictEntry(extentFn, idxTrait::GlobalNumThreadBlocks{}),
            DictEntry(strideFn, idxTrait::GlobalNumThreadBlocks{}))};

        constexpr auto withinThreadBlock = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::ThreadIdxInBlock{}),
            DictEntry(extentFn, idxTrait::NumThreadsInBlock{}),
            DictEntry(strideFn, idxTrait::NumThreadsInBlock{}))};
    } // namespace iter

} // namespace alpaka
