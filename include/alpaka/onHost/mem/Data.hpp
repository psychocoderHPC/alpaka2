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
        template<typename T_Elem, typename T_Idx, uint32_t T_dim>
        constexpr auto calculatePitchesFromExtents(Vec<T_Idx, T_dim> const& extent)
        {
            using VecType = Vec<T_Idx, T_dim>;
            VecType pitchBytes{};
            constexpr auto dim = VecType::dim();
            if constexpr(dim > 0)
                pitchBytes.back() = static_cast<T_Idx>(sizeof(T_Elem));
            if constexpr(dim > 1)
                for(T_Idx i = dim - 1; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }

        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, typename T_Idx, uint32_t T_dim>
        requires(T_dim >= 2)
        constexpr auto calculatePitches(Vec<T_Idx, T_dim> const& extent, T_Idx const& rowPitchBytes)
        {
            using VecType = Vec<T_Idx, T_dim>;
            VecType pitchBytes{};
            constexpr auto dim = VecType::dim();
            pitchBytes.back() = static_cast<T_Idx>(sizeof(T_Elem));
            if constexpr(dim > 1)
                pitchBytes[T_dim - 2u] = rowPitchBytes;
            if constexpr(dim > 2)
                for(T_Idx i = dim - 2; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }
    } // namespace mem

    template<typename T_BaseHandle, typename T_Type, typename T_Extents>
    struct Data : std::enable_shared_from_this<Data<T_BaseHandle, T_Type, T_Extents>>
    {
    public:
        Data(
            T_BaseHandle base,
            T_Type* data,
            T_Extents const& extents,
            T_Extents const& pitches,
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
        T_Extents m_pitches;
        std::function<void(T_Type*)> m_deleter;

        std::shared_ptr<Data> getSharedPtr()
        {
            return this->shared_from_this();
        }

        friend struct internal::Data;

        T_Extents getPitches() const
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

    template<typename T_BaseHandle, typename T_Type, typename T_Extents>
    struct GetApi::Op<onHost::Data<T_BaseHandle, T_Type, T_Extents>>
    {
        decltype(auto) operator()(auto&& data) const
        {
            return onHost::getApi(data.m_base);
        }
    };
} // namespace alpaka::internal
