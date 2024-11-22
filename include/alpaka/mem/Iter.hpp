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

namespace alpaka::onAcc
{
    namespace iter
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

            struct FrameCount
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[frame::count];
                }
            };

            struct FrameExtent
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[frame::extent];
                }
            };

            struct GridThreadIdx
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
                }
            };

            struct BlockIdx
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[layer::block].idx();
                }
            };

            struct BlockCount
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[layer::block].count();
                }
            };

            struct GridThreadCount
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[layer::block].count() * acc[layer::thread].count();
                }
            };

            struct ThreadCountInBlock
            {
                template<typename T_Acc>
                constexpr auto operator()(T_Acc const& acc) const
                {
                    return acc[layer::thread].count();
                }
            };

            struct ThreadIndexInBlock
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
                 * ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from __host__ __device__
                 * warning is popping up and generated code is wrong.
                 */
                template<typename T_Acc, typename T_DomainSpec, typename T_Traverse, typename T_IdxMapping>
                struct Op
                {
                    ALPAKA_FN_HOST_ACC constexpr auto operator()(
                        T_Acc const& acc,
                        T_DomainSpec const& domainSpec,
                        [[maybe_unused]] T_Traverse traverse,
                        T_IdxMapping idxMapping) const
                        requires std::is_same_v<ALPAKA_TYPE(idxMapping), idxLayout::Optimized>
                    {
                        auto adjIdxMapping = adjustMapping(acc, acc[object::api]);
                        auto const idxRange = domainSpec.getIdxRange(acc);
                        auto const threadSpace = domainSpec.getThreadSpace(acc);
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
                        T_DomainSpec const& domainSpec,
                        T_IdxMapping idxMapping) const
                    {
                        auto const idxRange = domainSpec.getIdxRange(acc);
                        auto const threadSpace = domainSpec.getThreadSpace(acc);
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

        template<typename T_WorkGroup, typename T_IdxRange>
        struct DomainSpec
        {
            constexpr DomainSpec(T_WorkGroup const threadGroup, T_IdxRange const idxRange)
                : m_threadGroup{threadGroup}
                , m_idxRange{idxRange}
            {
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
                requires(requires { std::declval<T_WorkGroup>().getThreadSpace(acc); })
            {
                return m_threadGroup.getThreadSpace(acc);
            }

            T_WorkGroup m_threadGroup;
            T_IdxRange m_idxRange;
        };

        template<typename T_Idx, typename T_Extent>
        struct WorkerGroup
        {
            constexpr WorkerGroup(T_Idx const& idxFn, T_Extent const& extentFn) : m_idxFn{idxFn}, m_extentFn{extentFn}
            {
            }

        private:
            template<typename T_ThreadGroup, typename T_IdxRange>
            friend struct DomainSpec;

            constexpr auto getThreadSpace([[maybe_unused]] auto const& acc) const
            {
                return ThreadSpace{m_idxFn, m_extentFn};
            }

            constexpr auto getThreadSpace(auto const& acc) const requires(requires {
                std::declval<T_Idx>()(acc);
                std::declval<T_Extent>()(acc);
            })
            {
                return ThreadSpace{m_idxFn(acc), m_extentFn(acc)};
            }

        private:
            T_Idx const m_idxFn;
            T_Extent const m_extentFn;
        };

    } // namespace iter

    namespace worker
    {
        constexpr auto threadsInGrid
            = iter::WorkerGroup{iter::idxTrait::GridThreadIdx{}, iter::idxTrait::GridThreadCount{}};
        constexpr auto blocksInGrid = iter::WorkerGroup{iter::idxTrait::BlockIdx{}, iter::idxTrait::BlockCount{}};
        constexpr auto threadsInBlock
            = iter::WorkerGroup{iter::idxTrait::ThreadIndexInBlock{}, iter::idxTrait::ThreadCountInBlock{}};
    } // namespace worker

    namespace range
    {
        constexpr auto dataExtent = iter::detail::IdxRangeFn{iter::idxTrait::DataExtent{}};
        constexpr auto frameCount = iter::detail::IdxRangeFn{iter::idxTrait::FrameCount{}};
        constexpr auto frameExtent = iter::detail::IdxRangeFn{iter::idxTrait::FrameExtent{}};
        constexpr auto threadInGrid = iter::detail::IdxRangeFn{iter::idxTrait::GridThreadCount{}};
        constexpr auto blocksInGrid = iter::detail::IdxRangeFn{iter::idxTrait::BlockCount{}};
        constexpr auto threadsInBlock = iter::detail::IdxRangeFn{iter::idxTrait::ThreadCountInBlock{}};
    } // namespace range

} // namespace alpaka::onAcc
