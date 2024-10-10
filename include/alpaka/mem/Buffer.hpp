/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/Handle.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/hostApi.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka
{
    template<typename T_Datahandle, typename T_Extents>
    struct Buffer
    {
    public:
        Buffer(T_Datahandle data, T_Extents const& extents) : m_data(std::move(data)), m_extents(extents)
        {
        }

        Buffer(T_Datahandle data) : m_data(std::move(data)), m_extents(m_data->m_extents)
        {
        }

        Buffer(Buffer const&) = default;
        Buffer(Buffer&&) = default;

        using type = typename T_Datahandle::element_type::type;

        consteval uint32_t dim() const
        {
            return T_Extents::dim();
        }

        auto getExtent() const
        {
            return m_extents;
        }

    private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        friend struct alpaka::internal::Memcpy;

        T_Datahandle m_data;
        T_Extents m_extents;

        friend struct alpaka::internal::Data;
        friend struct alpaka::internal::GetApi;
    };

    template<typename T_Datahandle>
    ALPAKA_FN_HOST_ACC Buffer(T_Datahandle) -> Buffer<T_Datahandle, typename T_Datahandle::element_type::ExtentType>;

    namespace internal
    {
        template<typename... T_Args>
        struct Data::Op<alpaka::Buffer<T_Args...>>
        {
            decltype(auto) operator()(auto&& buffer) const
            {
                return alpaka::data(buffer.m_data);
            }
        };

        template<typename... T_Args>
        struct GetApi::Op<alpaka::Buffer<T_Args...>>
        {
            decltype(auto) operator()(auto&& buffer) const
            {
                return alpaka::getApi(buffer.m_data);
            }
        };
    } // namespace internal
} // namespace alpaka
