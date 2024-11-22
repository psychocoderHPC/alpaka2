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
#include "alpaka/mem/FlatIdxContainer.hpp"
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/mem/TiledIdxContainer.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka::onAcc::iter
{
    namespace idxLayout
    {
        struct Strided
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

        struct Optimized
        {
        };

    } // namespace idxLayout

    constexpr auto Strided = idxLayout::Strided{};
    constexpr auto Contigious = idxLayout::Contigious{};
    constexpr auto Optimized = idxLayout::Optimized{};

    namespace traverse
    {
        struct Flat
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxMapper,
                concepts::CVector auto const& cSelect)
            {
                return FlatIdxContainer{idxRange, threadSpace, idxMapper, cSelect};
            }
        };

        struct Tiled
        {
            ALPAKA_FN_HOST_ACC static constexpr auto make(
                auto const& idxRange,
                auto const& threadSpace,
                auto const& idxMapper,
                concepts::CVector auto const& cSelect)
            {
                return TiledIdxContainer{idxRange, threadSpace, idxMapper, cSelect};
            }
        };
    } // namespace traverse

    constexpr auto flat = traverse::Flat{};
    constexpr auto tiled = traverse::Tiled{};

    namespace trait
    {
        template<typename T>
        struct IsIdxMapping : std::false_type
        {
        };

        template<>
        struct IsIdxMapping<idxLayout::Strided> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<idxLayout::Optimized> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<idxLayout::Contigious> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isIdxMapping_v = IsIdxMapping<T>::value;

        template<typename T>
        struct IsIdxTraversing : std::false_type
        {
        };

        template<>
        struct IsIdxTraversing<traverse::Flat> : std::true_type
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
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count] * acc[frame::extent];
            }
        };

        struct DataFrameCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count];
            }
        };

        struct DataFrameExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::extent];
            }
        };

        struct GlobalThreadIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
            }
        };

        struct GlobalThreadBlockIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].idx();
            }
        };

        struct GlobalNumThreadBlocks
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].count();
            }
        };

        struct GlobalNumThreads
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].count() * acc[layer::thread].count();
            }
        };

        struct NumThreadsInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::thread].count();
            }
        };

        struct ThreadIdxInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
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
                    return idxLayout::Strided{};
                }
            };
        };

        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Cpu) const
            {
                return idxLayout::Contigious{};
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
                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    [[maybe_unused]] T_Traverse traverse,
                    T_IdxMapping idxMapping) const
                    requires std::is_same_v<ALPAKA_TYPE(idxMapping), idxLayout::Optimized>
                {
                    auto adjIdxMapping = adjustMapping(acc, acc[object::api]);
                    auto const idxRange = rangeOps.getIdxRange(acc);
                    auto const threadSpace = rangeOps.getThreadSpace(acc);
                    return T_Traverse::make(
                        idxRange,
                        threadSpace,
                        adjIdxMapping,
                        iotaCVec<
                            typename ALPAKA_TYPE(idxRange.distance())::type,
                            ALPAKA_TYPE(idxRange.distance())::dim()>());
                }

                ALPAKA_FN_HOST_ACC constexpr auto operator()(
                    T_Acc const& acc,
                    T_RangeOps rangeOps,
                    T_IdxMapping idxMapping) const
                {
                    auto const idxRange = rangeOps.getIdxRange(acc);
                    auto const threadSpace = rangeOps.getThreadSpace(acc);
                    return T_Traverse::make(
                        idxRange,
                        threadSpace,
                        idxMapping,
                        iotaCVec<
                            typename ALPAKA_TYPE(idxRange.distance())::type,
                            ALPAKA_TYPE(idxRange.distance())::dim()>());
                }
            };
        };
    } // namespace internal

    namespace detail
    {
        template<typename T_IdxFn, typename T_ExtentFn>
        struct ThreadGroupFn
        {
            constexpr ThreadGroupFn(T_IdxFn const& idxFn, T_ExtentFn const& extentFn)
                : m_idxFn{idxFn}
                , m_extentFn{extentFn}
            {
            }

            constexpr auto getThreadSpace(auto const& acc) const
            {
                return ThreadSpace{m_idxFn(acc), m_extentFn(acc)};
            }

        private:
            T_IdxFn const m_idxFn;
            T_ExtentFn const m_extentFn;
        };

        template<typename T_ExtentFn>
        struct IdxRangeFn
        {
            constexpr IdxRangeFn(T_ExtentFn const& extentFn) : m_extentFn{extentFn}
            {
            }

            constexpr auto getIdxRange(auto const& acc) const
            {
                return IdxRange{m_extentFn(acc)};
            }

        private:
            T_ExtentFn const m_extentFn;
        };
    } // namespace detail

    template<typename T_ThreadGroup, typename T_IdxRange>
    struct DomainSpec
    {
        constexpr DomainSpec(T_ThreadGroup const& threadGroup, T_IdxRange const& idxRange)
            : m_threadGroup{threadGroup}
            , m_idxRange{idxRange}
        {
        }

        constexpr auto over(auto const& idxRange) const
        {
            return DomainSpec<T_ThreadGroup, ALPAKA_TYPE(idxRange)>{m_threadGroup, idxRange};
        }

    private:
        friend internal::MakeIter;

        constexpr auto getIdxRange(auto const& acc) const
        {
            return m_idxRange;
        }

        constexpr auto getIdxRange(auto const& acc) const
            requires(requires { std::declval<T_IdxRange>().getIdxRange(acc); })
        {
            return m_idxRange.getIdxRange(acc);
        }

        constexpr auto getThreadSpace(auto const& acc) const
        {
            return m_threadGroup;
        }

        constexpr auto getThreadSpace(auto const& acc) const
            requires(requires { std::declval<T_ThreadGroup>().getThreadSpace(acc); })
        {
            return m_threadGroup.getThreadSpace(acc);
        }

        T_ThreadGroup m_threadGroup;
        T_IdxRange m_idxRange;
    };

    constexpr auto gridThreads = detail::ThreadGroupFn{idxTrait::GlobalThreadIdx{}, idxTrait::GlobalNumThreads{}};
    constexpr auto blockThreads = detail::ThreadGroupFn{idxTrait::ThreadIdxInBlock{}, idxTrait::NumThreadsInBlock{}};

    constexpr auto overDataRange = DomainSpec{gridThreads, detail::IdxRangeFn{idxTrait::DataExtent{}}};

    constexpr auto overDataFrames = DomainSpec{gridThreads, detail::IdxRangeFn{idxTrait::DataFrameCount{}}};

    constexpr auto withinDataFrame = DomainSpec{blockThreads, detail::IdxRangeFn{idxTrait::DataFrameExtent{}}};

    constexpr auto overThreadRange = DomainSpec{gridThreads, detail::IdxRangeFn{idxTrait::GlobalNumThreads{}}};

    constexpr auto overThreadBlocks = DomainSpec{gridThreads, detail::IdxRangeFn{idxTrait::GlobalNumThreadBlocks{}}};

    constexpr auto withinThreadBlock = DomainSpec{blockThreads, detail::IdxRangeFn{idxTrait::NumThreadsInBlock{}}};

} // namespace alpaka::onAcc::iter
