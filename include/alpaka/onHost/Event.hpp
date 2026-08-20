/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/EventPolicyList.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/internal/interface.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>

namespace alpaka::onHost
{
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct Device;

    template<typename T_Device, alpaka::concepts::EventPolicyList T_EventPolicyList = EventPolicyList<>>
    struct Event;

    template<
        alpaka::concepts::Api T_Api,
        alpaka::concepts::DeviceKind T_DeviceKind,
        alpaka::concepts::EventPolicyList T_EventPolicyList>
    struct Event<Device<T_Api, T_DeviceKind>, T_EventPolicyList>
    {
    private:
        using DeviceInterface = Device<T_Api, T_DeviceKind>;
        using EventHandle = ALPAKA_TYPEOF(
            internal::MakeEvent::Op<ALPAKA_TYPEOF(*std::declval<DeviceInterface>().get()), T_EventPolicyList>{}(
                *std::declval<DeviceInterface>().get(),
                std::declval<T_EventPolicyList const&>()));

        EventHandle m_event;
        [[no_unique_address]] T_EventPolicyList m_policies;

    public:
        using element_type = typename EventHandle::element_type;

        template<typename T_Event>
        Event(Handle<T_Event>&& event, T_EventPolicyList const& policies)
            : m_event{std::forward<Handle<T_Event>>(event)}
            , m_policies{policies}
        {
        }

        auto* get() const
        {
            return m_event.get();
        }

        constexpr auto getApi() const
        {
            return alpaka::internal::getApi(*m_event.get());
        }

        constexpr T_EventPolicyList const& getPolicyList() const
        {
            return m_policies;
        }

        constexpr alpaka::concepts::Timing auto getTiming() const
        {
            return m_policies.getTiming();
        }

        [[nodiscard]] std::string getName() const
        {
            return alpaka::internal::GetName::Op<std::decay_t<decltype(*m_event.get())>>{}(*m_event.get());
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return internal::getNativeHandle(*m_event.get());
        }

        bool operator==(Event const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Event const& other) const
        {
            return this->get() != other.get();
        }

        /** Get the device of this event
         *
         * @return the device of this event
         */
        auto getDevice() const
        {
            return Device<T_Api, T_DeviceKind>{internal::getDevice(*m_event.get())};
        }

        bool isComplete() const
        {
            return internal::isEventComplete(*m_event.get());
        }
    };

    template<typename T_Event, alpaka::concepts::EventPolicyList T_EventPolicyList>
    Event(Handle<T_Event>&&, T_EventPolicyList) -> Event<
        Device<
            ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Event>())),
            ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Event>()))>,
        T_EventPolicyList>;

    /** Return the elapsed time between two timing-enabled queue markers.
     *
     * Both events must have been recorded on timing-enabled queues belonging to
     * the same device. Timing-disabled events intentionally have no overload.
     * The returned duration is always calculated as `end - start`. The events do
     * not need to be chronologically ordered; the duration is negative if the end
     * marker is reached before the start marker.
     */
    template<
        alpaka::concepts::Api T_Api,
        alpaka::concepts::DeviceKind T_DeviceKind,
        alpaka::concepts::EventPolicyList T_StartPolicies,
        alpaka::concepts::EventPolicyList T_EndPolicies>
    requires(
        std::same_as<ALPAKA_TYPEOF(T_StartPolicies::getTiming()), timing::Enabled>
        && std::same_as<ALPAKA_TYPEOF(T_EndPolicies::getTiming()), timing::Enabled>)
    auto getElapsedTime(
        Event<Device<T_Api, T_DeviceKind>, T_StartPolicies> const& start,
        Event<Device<T_Api, T_DeviceKind>, T_EndPolicies> const& end) -> std::chrono::duration<double>
    {
        if(start.getDevice() != end.getDevice())
            throw std::invalid_argument{"Elapsed time requires events from the same device"};

        return internal::GetElapsedTime::Op<ALPAKA_TYPEOF(*start.get()), ALPAKA_TYPEOF(*end.get())>{}(
            *start.get(),
            *end.get());
    }

} // namespace alpaka::onHost
