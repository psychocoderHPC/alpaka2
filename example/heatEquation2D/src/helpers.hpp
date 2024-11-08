/* Copyright 2020 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>

template<typename T, typename U>
using const_match = std::conditional_t<std::is_const_v<T>, U const, U>;

//! Helper function to get a pointer to an element in a multidimensional buffer
//!
//! \tparam T type of the element
//! \tparam TIdx index type
//! \param ptr pointer to the buffer
//! \param idx index of the element
//! \param pitch pitch of the buffer
template<typename T, typename TIdx>
ALPAKA_FN_ACC T* getElementPtr(T* ptr, alpaka::Vec<TIdx, 2u> idx, alpaka::Vec<TIdx, 2u> pitch)
{
    return reinterpret_cast<T*>(
        reinterpret_cast<const_match<T, std::byte>*>(ptr) + idx[0] * pitch[0] + idx[1] * pitch[1]);
}
