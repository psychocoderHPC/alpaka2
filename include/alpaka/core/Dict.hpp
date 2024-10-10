/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdio>
#include <tuple>
#include <utility>

namespace alpaka
{
    // https://stackoverflow.com/a/64606884
    template<typename X, typename T_Tuple>
    struct Idx
    {
        static_assert(sizeof(T_Tuple) && false);
    };

    template<typename X, template<typename...> typename T_Tuple, typename... T>
    struct Idx<X, T_Tuple<T...>>
    {
        template<std::size_t... idx>
        static constexpr ssize_t find_idx(std::index_sequence<idx...>)
        {
            return -1 + static_cast<ssize_t>(((std::is_same<X, typename T::KeyType>::value ? idx + 1 : 0) + ...));
        }

    public:
        static constexpr ssize_t value = find_idx(std::index_sequence_for<T...>{});
    };

    template<typename X, template<typename...> typename T_Tuple>
    class Idx<X, T_Tuple<>>
    {
        static constexpr ssize_t find_idx(std::index_sequence<>)
        {
            return -1;
        }

    public:
        static constexpr ssize_t value = find_idx(std::index_sequence_for<>{});
    };

    template<typename T_Key, typename T_Tuple>
    inline consteval ssize_t idx(T_Tuple&& t, T_Key const& key = T_Key{})
    {
        constexpr auto idx = Idx<T_Key, std::decay_t<T_Tuple>>::value;
        return idx;
    }

    template<typename T_Key, typename T_Tuple>
    inline constexpr bool hasTag(T_Tuple&& t, T_Key const& key = T_Key{})
    {
        constexpr auto idx = Idx<T_Key, std::decay_t<T_Tuple>>::value;
        return idx != -1;
    }

    template<typename T_Key, typename T_Tuple>
    inline constexpr auto& getTag(T_Tuple&& t, T_Key const& key = T_Key{})
    {
        constexpr auto idx = Idx<T_Key, std::decay_t<T_Tuple>>::value;
        static_assert(idx != -1, "Member in dict missing!");
        static_assert(idx < std::tuple_size_v<std::decay_t<T_Tuple>>, "index out of range!");
        return std::get<idx>(std::forward<T_Tuple>(t)).value;
    }

    template<typename T_Key, typename T_Value>
    struct DictEntry
    {
        using KeyType = T_Key;
        using ValueType = T_Value;

        constexpr DictEntry(T_Key const, T_Value const& v) : value{v}
        {
        }

        T_Value value;
    };

    namespace trait
    {
        template<typename T_Object, typename T_Sfinae = void>
        struct ToDictEntry
        {
            template<typename T>
            static constexpr auto get(T&& data)
            {
                return std::forward<T>(data);
            }
        };
    } // namespace trait

    template<typename... T_DictEntry>
    struct Dict
    {
        static_assert(sizeof...(T_DictEntry) && false);
    };

    template<typename... T_Keys, typename... T_Values>
    struct Dict<DictEntry<T_Keys, T_Values>...> : std::tuple<DictEntry<T_Keys, T_Values>...>
    {
        constexpr Dict(std::tuple<DictEntry<T_Keys, T_Values>...> const& data)
            : std::tuple<DictEntry<T_Keys, T_Values>...>{data}
        {
        }

        constexpr Dict(DictEntry<T_Keys, T_Values> const&... dictEntries)
            : std::tuple<DictEntry<T_Keys, T_Values>...>{dictEntries...}
        {
        }

        constexpr Dict(Dict const&) = default;
        constexpr Dict(Dict&&) = default;

        constexpr decltype(auto) operator[](auto const tag) const
        {
            return getTag(*this, tag);
        }

        constexpr decltype(auto) operator[](auto const tag)
        {
            return getTag(*this, tag);
        }
    };

    // type deduction guide
    template<typename... T_Keys, typename... T_Values>
    ALPAKA_FN_HOST_ACC Dict(std::tuple<DictEntry<T_Keys, T_Values>...> const&) -> Dict<DictEntry<T_Keys, T_Values>...>;

    template<typename... T_Keys, typename... T_Values>
    ALPAKA_FN_HOST_ACC Dict(DictEntry<T_Keys, T_Values> const&...) -> Dict<DictEntry<T_Keys, T_Values>...>;

} // namespace alpaka

namespace std
{
    template<typename... T_Keys, typename... T_Values>
    struct tuple_size<alpaka::Dict<alpaka::DictEntry<T_Keys, T_Values>...>>
    {
        static constexpr std::size_t value = sizeof...(T_Keys);
    };

    template<std::size_t I, typename... T_Keys, typename... T_Values>
    struct tuple_element<I, alpaka::Dict<alpaka::DictEntry<T_Keys, T_Values>...>>
    {
        using type = decltype(std::get<I>(std::declval<std::tuple<alpaka::DictEntry<T_Keys, T_Values>...>>()));
    };
} // namespace std

namespace alpaka
{

    template<std::size_t... idx0, std::size_t... idx1, typename T_Dict0, typename T_Dict1>
    constexpr auto joinDictHelper(
        std::index_sequence<idx0...>,
        std::index_sequence<idx1...>,
        T_Dict0 dict0,
        T_Dict1 dict1)
    {
        return Dict{std::get<idx0>(dict0)..., std::get<idx1>(dict1)...};
    }

    template<typename... T_Entries0, typename... T_Entries1>
    constexpr auto joinDict(Dict<T_Entries0...> dict0, Dict<T_Entries1...> dict1)
    {
        return joinDictHelper(
            std::index_sequence_for<T_Entries0...>{},
            std::index_sequence_for<T_Entries1...>{},
            dict0,
            dict1);
    }
} // namespace alpaka
