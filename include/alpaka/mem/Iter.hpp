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
        static constexpr auto adjust(
            concepts::Vector auto const& strideVec,
            concepts::Vector auto const& extentVec,
            concepts::Vector auto const& firstVec,
            concepts::Vector auto const& threadCount)
        {
            return std::make_tuple(firstVec * strideVec, extentVec, threadCount * strideVec);
        }
    };

    struct Contigious
    {
        static constexpr auto adjust(
            concepts::Vector auto const& strideVec,
            concepts::Vector auto const& extentVec,
            concepts::Vector auto const& firstVec,
            concepts::Vector auto const& threadCount)
        {
            auto numElements = core::divCeil(extentVec, threadCount * strideVec);
            auto first = firstVec * numElements * strideVec;
            return std::make_tuple(first, extentVec.min(first + numElements * strideVec), strideVec);
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

    template<typename T_IdxRange, typename T_ThreadSpace, typename T_IdxMapperFn, concepts::CVector T_CSelect>
    class IndexContainer : private T_IdxMapperFn
    {
        void _()
        {
            static_assert(std::ranges::forward_range<IndexContainer>);
        }

    public:
        using IdxType = typename T_IdxRange::IdxType;
        static constexpr uint32_t dim = T_IdxRange::dim();
        using IdxVecType = Vec<IdxType, dim>;

        ALPAKA_FN_ACC inline IndexContainer(
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
            friend class IndexContainer;

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
            friend class IndexContainer;
            friend class const_iterator_end;

            static constexpr uint32_t iterDim = T_CSelect::dim();
            using IterIdxVecType = Vec<IdxType, iterDim>;

            void _()
            {
                static_assert(std::forward_iterator<const_iterator>);
            }

            constexpr const_iterator(
                concepts::Vector auto const stride,
                concepts::Vector auto const extent,
                concepts::Vector auto const first)
                : m_current{first}
                , m_stride{stride.select(T_CSelect{})}
                , m_extent{extent.select(T_CSelect{})}
                , m_first(first.select(T_CSelect{}))
            {
                // range check required for 1 dimensional iterators
                if constexpr(iterDim > 1u)
                {
                    // invalidate current if one dimension is out of range.
                    bool isIndexValid = true;
                    for(uint32_t d = 1u; d < iterDim; ++d)
                        isIndexValid = isIndexValid && (first[d] < extent[d]);
                    if(!isIndexValid)
                        m_current[T_CSelect{}[0]] = m_extent[0];
                }

                // std::cout << "const iter " << m_current << m_extent << m_stride << std::endl;
            }

            ALPAKA_FN_ACC constexpr IdxType slowCurrent() const
            {
                return m_current[T_CSelect{}[0]];
            }

        public:
            constexpr IdxVecType operator*() const
            {
                return m_current;
            }

            // pre-increment the iterator
            ALPAKA_FN_ACC inline const_iterator& operator++()
            {
                for(uint32_t d = 0; d < iterDim; ++d)
                {
                    uint32_t const idx = iterDim - 1u - d;
                    m_current[T_CSelect{}[idx]] += m_stride[idx];
                    if constexpr(iterDim != 1u)
                    {
                        if(idx >= 1u && m_current[T_CSelect{}[idx]] >= m_extent[idx])
                        {
                            m_current[T_CSelect{}[idx]] = m_first[idx];
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
                return (slowCurrent() >= *other);
            }

            constexpr bool operator!=(const_iterator_end const& other) const
            {
                return not(*this == other);
            }

        private:
            // modified by the pre/post-increment operator
            IdxVecType m_current;
            // non-const to support iterator copy and assignment
            IterIdxVecType m_stride;
            IterIdxVecType m_extent;
            iter::detail::ReducedVector<IdxType, iterDim> m_first;
        };

        ALPAKA_FN_ACC inline const_iterator begin() const
        {
            auto [first, extent, stride] = this->adjust(
                m_idxRange.m_stride,
                m_idxRange.distance(),
                m_threadSpace.m_threadIdx,
                m_threadSpace.m_threadCount);
            return const_iterator(stride, m_idxRange.m_begin + extent, m_idxRange.m_begin + first);
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
            return IndexContainer<T_IdxRange, T_ThreadSpace, T_IdxMapperFn, ALPAKA_TYPE(iterDir)>(
                m_idxRange,
                m_threadSpace,
                T_IdxMapperFn{});
        }

    private:
        T_IdxRange m_idxRange;
        T_ThreadSpace m_threadSpace;
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
    ALPAKA_TAG(threadCountFn);

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
            /* create iterator
             *
             * ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from __host__ __device__ warning
             * is popping up and generated code is wrong.
             */
            template<typename T_Acc, typename T_RangeOps, typename T_IdxMapping>
            struct Op
            {
                ALPAKA_NO_HOST_ACC_WARNING
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
                    static_assert(
                        std::is_same_v<typename ALPAKA_TYPE(offset)::type, typename ALPAKA_TYPE(extent)::type>);

                    auto adjIdxMapping = adjustMapping(acc, acc[object::api]);
                    return IndexContainer{
                        IdxRange{offset, offset + extent, ALPAKA_TYPE(offset)::all(1u)},
                        ThreadSpace{rangeOps[firstFn](acc), rangeOps[threadCountFn](acc)},
                        adjIdxMapping,
                        iotaCVec<typename ALPAKA_TYPE(extent)::type, ALPAKA_TYPE(extent)::dim()>()};
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping,
                    concepts::Vector auto const& offset,
                    concepts::Vector auto const& extent) const
                {
                    static_assert(std::is_same_v<ALPAKA_TYPE(offset), ALPAKA_TYPE(extent)>);
                    return IndexContainer{
                        IdxRange{offset, offset + extent, ALPAKA_TYPE(offset)::all(1u)},
                        ThreadSpace{rangeOps[firstFn](acc), rangeOps[threadCountFn](acc)},
                        idxMapping,
                        iotaCVec<typename ALPAKA_TYPE(extent)::type, ALPAKA_TYPE(extent)::dim()>()};
                }
            };
        };
    } // namespace internal

    /**
     * ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from __host__ __device__ warning
     * is popping up and generated code is wrong.
     * @{
     */
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

    /** @} */

    namespace iter
    {
        constexpr auto overDataRange = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadIdx{}),
            DictEntry(extentFn, idxTrait::DataExtent{}),
            DictEntry(threadCountFn, idxTrait::GlobalNumThreads{}))};

        constexpr auto overDataFrames = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadBlockIdx{}),
            DictEntry(extentFn, idxTrait::DataFrameCount{}),
            DictEntry(threadCountFn, idxTrait::GlobalNumThreadBlocks{}))};

        constexpr auto withinDataFrame = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::ThreadIdxInBlock{}),
            DictEntry(extentFn, idxTrait::DataFrameExtent{}),
            DictEntry(threadCountFn, idxTrait::NumThreadsInBlock{}))};

        constexpr auto overThreadRange = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadIdx{}),
            DictEntry(extentFn, idxTrait::GlobalNumThreads{}),
            DictEntry(threadCountFn, idxTrait::GlobalNumThreads{}))};

        constexpr auto overThreadBlocks = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::GlobalThreadBlockIdx{}),
            DictEntry(extentFn, idxTrait::GlobalNumThreadBlocks{}),
            DictEntry(threadCountFn, idxTrait::GlobalNumThreadBlocks{}))};

        constexpr auto withinThreadBlock = Dict{std::make_tuple(
            DictEntry(firstFn, idxTrait::ThreadIdxInBlock{}),
            DictEntry(extentFn, idxTrait::NumThreadsInBlock{}),
            DictEntry(threadCountFn, idxTrait::NumThreadsInBlock{}))};
    } // namespace iter

} // namespace alpaka
