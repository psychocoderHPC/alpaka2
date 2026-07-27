/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/internal/interface.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>

namespace alpaka::onHost
{
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct Device;

    template<typename T_Device, alpaka::concepts::Timing T_Timing = timing::Disabled>
    struct Event;

    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind, alpaka::concepts::Timing T_Timing>
    struct Event<Device<T_Api, T_DeviceKind>, T_Timing>
    {
    private:
        using DeviceInterface = Device<T_Api, T_DeviceKind>;
        using EventHandle = ALPAKA_TYPEOF(
            internal::MakeEvent::Op<ALPAKA_TYPEOF(*std::declval<DeviceInterface>().get()), T_Timing>{}(
                *std::declval<DeviceInterface>().get(),
                T_Timing{}));

        EventHandle m_event;

    public:
        using element_type = typename EventHandle::element_type;

        template<typename T_Event>
        Event(Handle<T_Event>&& event, T_Timing) : m_event{std::forward<Handle<T_Event>>(event)}
        {
        }

        template<typename T_Event>
        Event(Handle<T_Event>&& event) requires std::same_as<T_Timing, timing::Disabled>
            : Event{std::forward<Handle<T_Event>>(event), timing::disabled}
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

        constexpr alpaka::concepts::Timing auto getTiming() const
        {
            return T_Timing{};
        }

        std::string getName() const
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
            return alpaka::onHost::internal::isEventComplete(*m_event.get());
        }
    };

    template<typename T_Event, alpaka::concepts::Timing T_Timing>
    Event(Handle<T_Event>&&, T_Timing) -> Event<
        Device<
            ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Event>())),
            ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Event>()))>,
        T_Timing>;

    template<typename T_Event>
    Event(Handle<T_Event>&&) -> Event<Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Event>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Event>()))>>;

    /** Return the elapsed time between two timing-enabled queue markers.
     *
     * Both events must have been recorded on timing-enabled queues belonging to
     * the same device. Timing-disabled events intentionally have no overload.
     * The returned duration is always calculated as `end - start`. The events do
     * not need to be chronologically ordered; the duration is negative if the end
     * marker is reached before the start marker.
     */
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    auto getElapsedTime(
        Event<Device<T_Api, T_DeviceKind>, timing::Enabled> const& start,
        Event<Device<T_Api, T_DeviceKind>, timing::Enabled> const& end) -> std::chrono::duration<double>
    {
        if(start.getDevice() != end.getDevice())
            throw std::invalid_argument{"Elapsed time requires events from the same device"};

        return internal::GetElapsedTime::Op<ALPAKA_TYPEOF(*start.get()), ALPAKA_TYPEOF(*end.get())>{}(
            *start.get(),
            *end.get());
    }

} // namespace alpaka::onHost
