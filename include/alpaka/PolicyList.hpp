/* Copyright 2026 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/meta/TypeListOps.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/unused.hpp"

#include <concepts>
#include <tuple>
#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        /** Identifies compile-time policy tags.
         *
         * Specialize this trait with std::true_type for an application-defined policy tag. A policy tag must also be
         * default-initializable to satisfy alpaka::concepts::Policy.
         *
         * @tparam T_Policy Type to inspect.
         */
        template<typename T_Policy>
        struct IsPolicy : std::false_type
        {
        };
    } // namespace trait

    /** Whether a type is registered as a compile-time policy tag. */
    template<typename T_Policy>
    constexpr bool isPolicy_v = trait::IsPolicy<std::remove_cvref_t<T_Policy>>::value;

    namespace concepts
    {
        /** A registered, default-initializable compile-time policy tag. */
        template<typename T_Policy>
        concept Policy = isPolicy_v<T_Policy> && std::default_initializable<std::remove_cvref_t<T_Policy>>;
    } // namespace concepts

    /** Collection of compile-time policy tags.
     *
     * Policy values carry no runtime state. Derived bundles can use search() with a category predicate to select one
     * policy and provide a category-specific default.
     *
     * @tparam T_Policies Registered policy tag types contained in the bundle.
     */
    template<concepts::Policy... T_Policies>
    struct PolicyList
    {
    protected:
        using Args = std::tuple<T_Policies...>;

        /** Select the policy matching a category predicate.
         *
         * @param predicate Callable returning true for policies in the requested category.
         * @param defaultPolicy Policy returned when the bundle contains no match.
         * @return The matching policy, or defaultPolicy when no policy matches.
         */
        template<typename T_Predicate, concepts::Policy T_DefaultPolicy>
        static constexpr auto search(T_Predicate predicate, T_DefaultPolicy defaultPolicy)
        {
            auto const matches = meta::filter(predicate, Args{});
            static_assert(std::tuple_size_v<ALPAKA_TYPEOF(matches)> <= 1u, "Duplicate policy category.");

            if constexpr(std::tuple_size_v<ALPAKA_TYPEOF(matches)> == 0u)
                return defaultPolicy;
            else
                return meta::Front<ALPAKA_TYPEOF(matches)>{};
        }

    public:
        /** Construct a bundle from compile-time policy tags.
         *
         * @param policies Policy tags represented by this bundle's type.
         */
        constexpr PolicyList(T_Policies... policies)
        {
            alpaka::unused(policies...);
        }

        /** Check whether the bundle contains a policy type.
         *
         * @param policy Policy tag whose type is queried.
         * @return true if the exact policy type is present, otherwise false.
         */
        constexpr bool hasPolicy(concepts::Policy auto policy) const
        {
            return meta::Contains<Args, ALPAKA_TYPEOF(policy)>::value;
        }
    };
} // namespace alpaka
