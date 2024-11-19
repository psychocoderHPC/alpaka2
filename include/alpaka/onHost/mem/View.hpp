/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    template<typename T_Datahandle, typename T_Extents>
    struct View
    {
    public:
        View(T_Datahandle data, T_Extents const& extents) : m_data(std::move(data)), m_extents(extents)
        {
        }

        View(T_Datahandle data) : m_data(std::move(data)), m_extents(m_data->m_extents)
        {
        }

        View(View const&) = default;
        View(View&&) = default;

        View& operator=(View const&) = default;

        using type = typename T_Datahandle::element_type::type;

        consteval uint32_t dim() const
        {
            return T_Extents::dim();
        }

        auto getExtents() const
        {
            return m_extents;
        }

        auto getPitches() const
        {
            return m_data->getPitches();
        }

        decltype(auto) data()
        {
            return onHost::data(m_data);
        }

        decltype(auto) data() const
        {
            return onHost::data(m_data);
        }

        auto getMdSpan() const
        {
            auto* ptr = onHost::data(m_data);
            return alpaka::MdSpan{ptr, m_data->getExtents(), m_data->getPitches()};
        }

    private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        friend struct internal::Memcpy;

        T_Datahandle m_data;
        T_Extents m_extents;

        friend struct internal::Data;
        friend struct alpaka::internal::GetApi;
    };

    template<typename T_Datahandle>
    ALPAKA_FN_HOST_ACC View(T_Datahandle) -> View<T_Datahandle, typename T_Datahandle::element_type::ExtentType>;

    namespace internal
    {
        template<typename... T_Args>
        struct Data::Op<View<T_Args...>>
        {
            decltype(auto) operator()(auto&& buffer) const
            {
                return onHost::data(buffer.m_data);
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename... T_Args>
    struct GetApi::Op<onHost::View<T_Args...>>
    {
        decltype(auto) operator()(auto&& buffer) const
        {
            return onHost::getApi(buffer.m_data);
        }
    };
} // namespace alpaka::internal
