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
#include "alpaka/mem/LinearizedIdxContainer.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/mem/TilingIdxContainer.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka::iter
{
    namespace layout
    {
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

        constexpr auto stride = Stride{};

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

        constexpr auto contigious = Contigious{};

        struct Optimize
        {
        };

        constexpr auto optimize = Optimize{};
    } // namespace layout

    namespace traverse
    {
        struct Linearized
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxMapper,
                concepts::CVector auto const& cSelect)
            {
                return LinearizedIdxContainer{idxRange, threadSpace, idxMapper, cSelect};
            }
        };

        constexpr auto linearized = Linearized{};

        struct Tiled
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxMapper,
                concepts::CVector auto const& cSelect)
            {
                return TilingIdxContainer{idxRange, threadSpace, idxMapper, cSelect};
            }
        };

        constexpr auto tiled = Tiled{};
    } // namespace traverse

    namespace trait
    {
        template<typename T>
        struct IsIdxMapping : std::false_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Stride> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Optimize> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Contigious> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isIdxMapping_v = IsIdxMapping<T>::value;

        template<typename T>
        struct IsIdxTraversing : std::false_type
        {
        };

        template<>
        struct IsIdxTraversing<traverse::Linearized> : std::true_type
        {
        };

        template<>
        struct IsIdxTraversing<traverse::Tiled> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isIdxTraversing_v = IsIdxTraversing<T>::value;

    } // namespace trait

    namespace concepts
    {
        template<typename T>
        concept IdxMapping = trait::isIdxMapping_v<T>;

        template<typename T>
        concept IdxTraversing = trait::isIdxTraversing_v<T>;
    } // namespace concepts

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
                    return layout::Stride{};
                }
            };
        };

        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Cpu) const
            {
                return layout::Contigious{};
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
            template<typename T_Acc, typename T_RangeOps, typename T_Traverse, typename T_IdxMapping>
            struct Op
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_Traverse traverse,
                    T_IdxMapping idxMapping) const
                {
                    return (*this)(acc, rangeOps, traverse, idxMapping, rangeOps[extentFn](acc));
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_Traverse traverse,
                    T_IdxMapping idxMapping,
                    alpaka::concepts::Vector auto const& extent) const
                {
                    return (*this)(acc, rangeOps, traverse, idxMapping, ALPAKA_TYPE(extent)::all(0), extent);
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    [[maybe_unused]] T_Traverse traverse,
                    T_IdxMapping idxMapping,
                    alpaka::concepts::Vector auto const& offset,
                    alpaka::concepts::Vector auto const& extent) const
                    requires std::is_same_v<ALPAKA_TYPE(idxMapping), layout::Optimize>
                {
                    static_assert(
                        std::is_same_v<typename ALPAKA_TYPE(offset)::type, typename ALPAKA_TYPE(extent)::type>);

                    auto adjIdxMapping = adjustMapping(acc, acc[object::api]);
                    return T_Traverse::make(
                        IdxRange{offset, offset + extent, ALPAKA_TYPE(offset)::all(1u)},
                        ThreadSpace{rangeOps[firstFn](acc), rangeOps[threadCountFn](acc)},
                        adjIdxMapping,
                        iotaCVec<typename ALPAKA_TYPE(extent)::type, ALPAKA_TYPE(extent)::dim()>());
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping,
                    alpaka::concepts::Vector auto const& offset,
                    alpaka::concepts::Vector auto const& extent) const
                {
                    static_assert(std::is_same_v<ALPAKA_TYPE(offset), ALPAKA_TYPE(extent)>);
                    return T_Traverse::make(
                        IdxRange{offset, offset + extent, ALPAKA_TYPE(offset)::all(1u)},
                        ThreadSpace{rangeOps[firstFn](acc), rangeOps[threadCountFn](acc)},
                        idxMapping,
                        iotaCVec<typename ALPAKA_TYPE(extent)::type, ALPAKA_TYPE(extent)::dim()>());
                }
            };
        };
    } // namespace internal

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

} // namespace alpaka::iter
