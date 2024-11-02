/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Handle.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/hostApi.hpp"
#include "alpaka/mem/MdSpan.hpp"

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

        auto getMdSpan() const
        {
            // import mdspan into alpaka::experimental namespace. see: https://eel.is/c++draft/mdspan.syn
            using std::experimental::default_accessor;
            using std::experimental::dextents;
            using std::experimental::extents;
            using std::experimental::layout_left;
            using std::experimental::layout_right;
            using std::experimental::layout_stride;
            using std::experimental::mdspan;
            // import submdspan as well, which is not standardized yet
            using std::experimental::full_extent;
            using std::experimental::submdspan;

            auto* ptr = reinterpret_cast<std::byte*>(data(m_data));
            auto ex = detail::makeExtents(m_extents, std::make_index_sequence<T_Extents::dim()>{});
            auto const strides = m_data->m_pitch;
            layout_stride::mapping<decltype(ex)> m{ex, strides.toStdArray()};
            return alpaka::MdSpan{
                mdspan<type, decltype(ex), layout_stride, detail::ByteIndexedAccessor<type>>{ptr, m}};
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
