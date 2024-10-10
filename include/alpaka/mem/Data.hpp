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

    template<typename T_BaseHandle, typename T_Type, typename T_Extents>
    struct Data : std::enable_shared_from_this<Data<T_BaseHandle, T_Type, T_Extents>>
    {
    public:
        Data(
            T_BaseHandle base,
            T_Type* data,
            T_Extents const& extents,
            T_Extents const& pitch,
            std::function<void(T_Type*)> deleter)
            : m_base(std::move(base))
            , m_data(data)
            , m_extents(extents)
            , m_pitch(pitch)
            , m_deleter(deleter)
        {
        }

        Data(Data const&) = default;
        Data(Data&&) = default;

        ~Data()
        {
            // NOTE: m_pMem is allowed to be a nullptr here.
            m_deleter(m_data);
        }

        using type = T_Type;
        using ExtentType = T_Extents;

        // private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        T_BaseHandle m_base;
        T_Type* m_data;
        T_Extents m_extents;
        T_Extents m_pitch;
        std::function<void(T_Type*)> m_deleter;

        std::shared_ptr<Data> getSharedPtr()
        {
            return this->shared_from_this();
        }

        friend struct alpaka::internal::Data;

        T_Type const* data() const
        {
            return m_data;
        }

        T_Type* data()
        {
            return m_data;
        }

        friend struct alpaka::internal::GetApi;
    };

    namespace internal
    {
        template<typename T_BaseHandle, typename T_Type, typename T_Extents>
        struct GetApi::Op<alpaka::Data<T_BaseHandle, T_Type, T_Extents>>
        {
            decltype(auto) operator()(auto&& data) const
            {
                return alpaka::getApi(data.m_base);
            }
        };
    } // namespace internal
} // namespace alpaka
