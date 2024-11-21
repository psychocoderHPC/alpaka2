/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace mem
    {
        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        constexpr auto calculatePitchesFromExtents(T_Vec const& extent)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            if constexpr(dim > 0)
                pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                for(type i = dim - 1; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }

        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        requires(T_Vec::dim() >= 2)
        constexpr auto calculatePitches(T_Vec const& extent, typename T_Vec::type const& rowPitchBytes)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                pitchBytes[dim - 2u] = rowPitchBytes;
            if constexpr(dim > 2)
                for(type i = dim - 2; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }
    } // namespace mem

    template<
        typename T_BaseHandle,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Vector T_Pitches>
    struct Data : std::enable_shared_from_this<Data<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
    public:
        Data(
            T_BaseHandle base,
            T_Type* data,
            T_Extents const& extents,
            T_Pitches const& pitches,
            std::function<void(T_Type*)> deleter)
            : m_base(std::move(base))
            , m_data(data)
            , m_extents(extents)
            , m_pitches(pitches)
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
        T_Pitches m_pitches;
        std::function<void(T_Type*)> m_deleter;

        std::shared_ptr<Data> getSharedPtr()
        {
            return this->shared_from_this();
        }

        friend struct internal::Data;

        T_Pitches getPitches() const
        {
            return m_pitches;
        }

        T_Extents getExtents() const
        {
            return m_extents;
        }

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
} // namespace alpaka::onHost

namespace alpaka::internal
{

    template<typename T_BaseHandle, typename T_Type, typename T_Extents, typename T_Pitches>
    struct GetApi::Op<onHost::Data<T_BaseHandle, T_Type, T_Extents, T_Pitches>>
    {
        decltype(auto) operator()(auto&& data) const
        {
            return onHost::getApi(data.m_base);
        }
    };
} // namespace alpaka::internal
