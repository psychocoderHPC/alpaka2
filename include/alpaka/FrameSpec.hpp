/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/ThreadSpec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>
#include <ostream>

namespace alpaka
{
    template<alpaka::concepts::Vector T_NumFrames, alpaka::concepts::Vector T_FrameExtent>
    struct FrameSpec
    {
        using type = typename T_NumFrames::type;

        consteval uint32_t dim() const
        {
            return T_FrameExtent::dim();
        }

        T_NumFrames m_numFrames;
        T_FrameExtent m_frameExtent;
        ThreadSpec<T_NumFrames, T_FrameExtent> m_threadSpec;

        FrameSpec(T_NumFrames const& numFrames, T_FrameExtent const& frameExtent)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numFrames, frameExtent)
        {
        }

        FrameSpec(T_NumFrames const& numFrames, T_FrameExtent const& frameExtent, T_FrameExtent const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numFrames, numThreads)
        {
        }

        FrameSpec(
            T_NumFrames const& numFrames,
            T_FrameExtent const& frameExtent,
            T_NumFrames numBlocks,
            T_FrameExtent const& numThreads)
            : m_numFrames(numFrames)
            , m_frameExtent(frameExtent)
            , m_threadSpec(numBlocks, numThreads)
        {
        }

        auto getThreadSpec() const
        {
            return m_threadSpec;
        }
    };

    template<alpaka::concepts::Vector T_NumFrames, alpaka::concepts::Vector T_FrameExtent>
    std::ostream& operator<<(std::ostream& s, FrameSpec<T_NumFrames, T_FrameExtent> const& d)
    {
        return s << "frames=" << d.m_numFrames << " frameExtent=" << d.m_frameExtent;
    }
} // namespace alpaka
