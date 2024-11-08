/* Copyright 2023 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/api.hpp"

#include <functional>
#include <tuple>
#include <utility>

#pragma once

namespace alpaka
{
    //! execute a callable for each active accelerator tag
    //
    // @param callable callable which can be invoked with an accelerator tag
    // @return disjunction of all invocation results
    //
    inline auto executeForEach(auto&& callable, auto const& tuple)
    {
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... api) { return (callable(api) || ...); }, tuple);
    }
#if 0
    inline auto executeForEachNoReturn(auto&& callable, auto const& tuple)
    {
        // Execute the callable once for each enabled accelerator.
        // Pass the tag as first argument to the callable.
        return std::apply([=](auto const&... api) { (callable(api), ...); }, tuple);
    }
#endif
} // namespace alpaka
