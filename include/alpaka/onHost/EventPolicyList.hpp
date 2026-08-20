/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onHost/QueuePolicyList.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

namespace alpaka::trait
{
    /** Identifies compile-time event policy tags.
     *
     * Specialize this trait with std::true_type to register an application-defined event policy. Every event policy is
     * also registered as a general policy through IsPolicy.
     *
     * @tparam T_Policy Type to inspect.
     */
    template<typename T_Policy>
    struct IsEventPolicy : std::false_type
    {
    };

    template<concepts::Timing T_Timing>
    struct IsEventPolicy<T_Timing> : std::true_type
    {
    };
} // namespace alpaka::trait

namespace alpaka
{
    /** Whether a type is registered as a compile-time event policy tag. */
    template<typename T_Policy>
    constexpr bool isEventPolicy_v = trait::IsEventPolicy<std::remove_cvref_t<T_Policy>>::value;

    namespace concepts
    {
        /** A registered, default-initializable compile-time event policy tag. */
        template<typename T_Policy>
        concept EventPolicy = isEventPolicy_v<T_Policy> && std::default_initializable<std::remove_cvref_t<T_Policy>>;
    } // namespace concepts

    namespace trait
    {
        template<concepts::EventPolicy T_Policy>
        // the requires is in-place since timing already has been defined as IsPolicy under QueuePolicyList.hpp
        requires(!concepts::QueuePolicy<T_Policy>)
        struct IsPolicy<T_Policy> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::onHost
{
    /** Collection of compile-time policies used to construct an event.
     *
     * Policies can be supplied in any order. At most one policy from each category may be present. Timing defaults to
     * timing::disabled when the timing policy is omitted.
     *
     * @tparam T_Policies Event policy tag types contained in the bundle.
     */
    template<::alpaka::concepts::EventPolicy... T_Policies>
    struct EventPolicyList : PolicyList<T_Policies...>
    {
        using Base = PolicyList<T_Policies...>;

        /** Construct an event policy bundle.
         *
         * @param policies Compile-time event policy tags.
         */
        constexpr EventPolicyList(T_Policies... policies) : Base{policies...}
        {
        }

        /** Return the selected timing policy, or timing::disabled if none was supplied. */
        static constexpr alpaka::concepts::Timing auto getTiming()
        {
            return Base::search(category::Timing{}, timing::disabled);
        }

        using Base::hasPolicy;
    };

    template<typename... T_Policies>
    EventPolicyList(T_Policies...) -> EventPolicyList<T_Policies...>;
} // namespace alpaka::onHost

namespace alpaka::concepts
{
    /** A specialization of onHost::EventPolicyList. */
    template<typename T_Policies>
    concept EventPolicyList = SpecializationOf<T_Policies, onHost::EventPolicyList>;
} // namespace alpaka::concepts
