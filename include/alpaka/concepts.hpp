/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/internal.hpp"

#include <concepts>
#include <string>

namespace alpaka
{
    namespace concepts
    {
        template<typename T>
        concept HasStaticName = requires(T t) {
            {
                internal::GetStaticName::Op<std::decay_t<T>>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasName = requires(T t) {
            {
                internal::GetName::Op<T>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasGet = requires(T t) { t.get(); };

        template<typename T>
        concept HasStaticDim = requires(T t) { T::dim(); };


        template<typename T>
        concept Api = requires(T t) { requires HasStaticName<T>; };

        template<typename T, unsigned int T_dim>
        concept Dim = requires { T::dim() == T_dim; };
    } // namespace concepts
} // namespace alpaka
