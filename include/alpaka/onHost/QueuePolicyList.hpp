/* Copyright 2026 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/PolicyList.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/utility.hpp"

namespace alpaka::trait
{
    /** Identifies compile-time queue policy tags.
     *
     * Specialize this trait with std::true_type to register an application-defined queue policy. Every queue policy is
     * also registered as a general policy through IsPolicy.
     *
     * @tparam T_Policy Type to inspect.
     */
    template<typename T_Policy>
    struct IsQueuePolicy : std::false_type
    {
    };

    template<alpaka::concepts::QueueKind T_QueueKind>
    struct IsQueuePolicy<T_QueueKind> : std::true_type
    {
    };

    template<alpaka::concepts::Timing T_Timing>
    struct IsQueuePolicy<T_Timing> : std::true_type
    {
    };
} // namespace alpaka::trait

namespace alpaka
{
    /** Whether a type is registered as a compile-time queue policy tag. */
    template<typename T_Policy>
    constexpr bool isQueuePolicy_v = trait::IsQueuePolicy<std::remove_cvref_t<T_Policy>>::value;

    namespace concepts
    {
        /** A registered, default-initializable compile-time queue policy tag. */
        template<typename T_Policy>
        concept QueuePolicy = isQueuePolicy_v<T_Policy> && std::default_initializable<std::remove_cvref_t<T_Policy>>;
    } // namespace concepts

    namespace trait
    {
        template<alpaka::concepts::QueuePolicy T_Policy>
        struct IsPolicy<T_Policy> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::onHost
{
    /** Collection of compile-time policies used to construct a queue.
     *
     * Policies can be supplied in any order. At most one policy from each category may be present. The queue kind
     * defaults to queueKind::nonBlocking and timing defaults to timing::disabled when their policies are omitted.
     *
     * @tparam T_Policies Queue policy tag types contained in the bundle.
     */
    template<alpaka::concepts::QueuePolicy... T_Policies>
    struct QueuePolicyList : PolicyList<T_Policies...>
    {
        using Base = PolicyList<T_Policies...>;

        /** Construct a queue policy bundle.
         *
         * @param policies Compile-time queue policy tags.
         */
        constexpr QueuePolicyList(T_Policies... policies) : Base{policies...}
        {
        }

        /** Return the selected queue kind, or queueKind::nonBlocking if none was supplied. */
        static constexpr alpaka::concepts::QueueKind auto getQueueKind()
        {
            return Base::search(category::QueueKind{}, queueKind::nonBlocking);
        }

        /** Return the selected timing policy, or timing::disabled if none was supplied. */
        static constexpr alpaka::concepts::Timing auto getTiming()
        {
            return Base::search(category::Timing{}, timing::disabled);
        }

        using Base::hasPolicy;
    };

    template<typename... T_Policies>
    QueuePolicyList(T_Policies...) -> QueuePolicyList<T_Policies...>;
} // namespace alpaka::onHost

namespace alpaka::concepts
{
    /** A specialization of onHost::QueuePolicyList. */
    template<typename T_Policies>
    concept QueuePolicyList = SpecializationOf<T_Policies, onHost::QueuePolicyList>;
} // namespace alpaka::concepts
