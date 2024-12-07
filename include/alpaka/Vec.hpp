/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/core/util.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <sstream>
#include <type_traits>

namespace alpaka
{
    /** Array storge for vector data
     *
     * This class is a workaround and is simply wrapping std::array. It is required because the dim in std::array
     * in the template signature is size_t. This produces template deduction issues for math::Vec if we sue
     * array as default storage without this wrapper.
     */
    template<typename T_Type, uint32_t T_dim>
    struct ArrayStorage : protected std::array<T_Type, T_dim>
    {
        using type = T_Type;
        using BaseType = std::array<T_Type, T_dim>;
        using BaseType::operator[];

        // constructor is required because exposing the array constructors does not work
        template<typename... T_Args>
        constexpr ArrayStorage(T_Args&&... args) : BaseType{std::forward<T_Args>(args)...}
        {
        }

        constexpr ArrayStorage(std::array<T_Type, T_dim> const& data) : BaseType{data}
        {
        }
    };

    namespace detail
    {
        struct GetValue
        {
            constexpr auto operator()(auto idx, auto value) const
            {
                return value;
            }
        };

        template<typename T, T... T_values>
        struct CVec
        {
            using Values = std::tuple<std::integral_constant<T, T_values>...>;
            using type = T;

            static consteval uint32_t dim()
            {
                return std::tuple_size_v<Values>;
            }

            constexpr T operator[](std::integral auto const idx) const
            {
                T result;
                using IdxType = ALPAKA_TYPEOF(idx);
                std::apply(
                    [&result, idx](auto const&... v)
                    {
                        IdxType i{0u};
                        /**
                         *  sizeof() is required to return a bool which is deferred in the evaluation
                         */
                        (void) (((idx == i++) && (result = ALPAKA_TYPEOF(v)::value, sizeof(IdxType))) || ...);
                    },
                    Values{});
                return result;
            }
        };


        template<typename T>
        struct IsTemplateSignatureStorage : std::false_type
        {
        };

        template<typename T_Type, T_Type... T_values>
        struct IsTemplateSignatureStorage<CVec<T_Type, T_values...>> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isTemplateSignatureStorage_v = IsTemplateSignatureStorage<T>::value;
    } // namespace detail

    template<typename T_Type, uint32_t T_dim, typename T_Storage = ArrayStorage<T_Type, T_dim>>
    struct Vec;

    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    struct Vec : private T_Storage
    {
        using Storage = T_Storage;
        using type = T_Type;
        using ParamType = type;

        using index_type = T_Type;
        using size_type = uint32_t;
        using rank_type = uint32_t;

        // universal vec used as fallback if T_Storage is holding the state in the template signature
        using UniVec = Vec<T_Type, T_dim>;

        /*Vecs without elements are not allowed*/
        static_assert(T_dim > 0u);

        constexpr Vec() = default;

        /** Initialize via a generator expression
         *
         * The generator must return the value for the corresponding index of the component which is passed to the
         * generator.
         */
        template<
            typename F,
            std::enable_if_t<std::is_invocable_v<F, std::integral_constant<uint32_t, 0u>>, uint32_t> = 0u>
        constexpr explicit Vec(F&& generator)
            : Vec(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_dim>{})
        {
        }

    private:
        template<typename F, uint32_t... Is>
        constexpr explicit Vec(F&& generator, std::integer_sequence<uint32_t, Is...>)
            : Storage{generator(std::integral_constant<uint32_t, Is>{})...}
        {
        }

    public:
        /** Constructor for N-dimensional vector
         *
         * @attention This constructor allows implicit casts.
         *
         * @param args value of each dimension, x,y,z,...
         *
         * A constexpr vector should be initialized with {} instead of () because at least
         * CUDA 11.6 has problems in cases where a compile time evaluation is required.
         * @code{.cpp}
         *   constexpr auto vec1 = Vec{ 1 };
         *   constexpr auto vec2 = Vec{ 1, 2 };
         *   //or explicit
         *   constexpr auto vec3 = Vec<int, 3u>{ 1, 2, 3 };
         *   constexpr auto vec4 = Vec<int, 3u>{ {1, 2, 3} };
         * @endcode
         */
        template<typename... T_Args, typename = std::enable_if_t<(std::is_convertible_v<T_Args, T_Type> && ...)>>
        constexpr Vec(T_Args... args) : Storage(static_cast<T_Type>(args)...)
        {
        }

        constexpr Vec(Vec const& other) = default;

        constexpr Vec(T_Storage const& other) : T_Storage{other}
        {
        }

        /** constructor allows changing the storage policy
         */
        template<typename T_OtherStorage>
        constexpr Vec(Vec<T_Type, T_dim, T_OtherStorage> const& other)
            : Vec([&](uint32_t const i) constexpr { return other[i]; })
        {
        }

        template<
            typename T_OtherType,
            typename T_OtherStorage,
            typename = std::enable_if_t<std::is_convertible_v<T_OtherType, T_Type>>>
        constexpr explicit Vec(Vec<T_OtherType, T_dim, T_OtherStorage> const& other)
            : Vec([&](uint32_t const i) constexpr { return static_cast<T_Type>(other[i]); })
        {
        }

        /** Allow static_cast / explicit cast to member type for 1D vector */
        template<uint32_t T_deferDim = T_dim, typename = typename std::enable_if<T_deferDim == 1u>::type>
        constexpr explicit operator type()
        {
            return (*this)[0];
        }

        static consteval uint32_t dim()
        {
            return T_dim;
        }

        /**
         * Creates a Vec where all dimensions are set to the same value
         *
         * @param value Value which is set for all dimensions
         * @return new Vec<...>
         */
        static constexpr auto all(T_Type const& value)
        {
            if constexpr(requires { detail::isTemplateSignatureStorage_v<T_Storage>; })
            {
                UniVec result([=](uint32_t const) { return value; });
                return result;
            }
            else
            {
                Vec result([=](uint32_t const) { return value; });
                return result;
            }
        }

        constexpr Vec toRT() const

        {
            return *this;
        }

        constexpr Vec revert() const
        {
            Vec invertedVec{};
            for(uint32_t i = 0u; i < T_dim; i++)
                invertedVec[T_dim - 1 - i] = (*this)[i];

            return invertedVec;
        }

        constexpr Vec& operator=(Vec const&) = default;

        constexpr Vec operator-() const
        {
            return Vec([this](uint32_t const i) constexpr { return -(*this)[i]; });
        }

/** assign operator
 * @{
 */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr Vec& operator op(Vec<T_Type, T_dim, T_OtherStorage> const& rhs)                                         \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op rhs[i]; })                                                 \
            {                                                                                                         \
                unWrapp((*this)[i]) op rhs[i];                                                                        \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op rhs[i];                                                                                 \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr Vec& operator op(T_Type const value)                                                                    \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op value; })                                                  \
            {                                                                                                         \
                unWrapp((*this)[i]) op value;                                                                         \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op value;                                                                                  \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }


        ALPAKA_VECTOR_ASSIGN_OP(+=)
        ALPAKA_VECTOR_ASSIGN_OP(-=)
        ALPAKA_VECTOR_ASSIGN_OP(/=)
        ALPAKA_VECTOR_ASSIGN_OP(*=)
        ALPAKA_VECTOR_ASSIGN_OP(=)

#undef ALPAKA_VECTOR_ASSIGN_OP

        /** @} */

        constexpr decltype(auto) operator[](uint32_t const idx)
        {
            return Storage::operator[](idx);
        }

        constexpr decltype(auto) operator[](uint32_t const idx) const
        {
            return Storage::operator[](idx);
        }

        /** named member access
         *
         * index -> name [0->x,1->y,2->z,3->w]
         * @{
         */
#define ALPAKA_NAMED_ARRAY_ACCESS(functionName, dimValue)                                                             \
    template<uint32_t T_deferDim = T_dim, std::enable_if_t<T_deferDim >= dimValue + 1u, int> = 0>                     \
    constexpr decltype(auto) functionName()                                                                           \
    {                                                                                                                 \
        return (*this)[T_dim - 1u - dimValue];                                                                        \
    }                                                                                                                 \
    template<uint32_t T_deferDim = T_dim, std::enable_if_t<T_deferDim >= dimValue + 1u, int> = 0>                     \
    constexpr decltype(auto) functionName() const                                                                     \
    {                                                                                                                 \
        return (*this)[T_dim - 1u - dimValue];                                                                        \
    }

        ALPAKA_NAMED_ARRAY_ACCESS(x, 0)
        ALPAKA_NAMED_ARRAY_ACCESS(y, 1)
        ALPAKA_NAMED_ARRAY_ACCESS(z, 2)
        ALPAKA_NAMED_ARRAY_ACCESS(w, 3)

#undef ALPAKA_NAMED_ARRAY_ACCESS

        /** @} */

        constexpr decltype(auto) back()
        {
            return (*this)[T_dim - 1u];
        }

        constexpr decltype(auto) back() const
        {
            return (*this)[T_dim - 1u];
        }

        /** Shrink the number of elements of a vector.
         *
         * Highest indices kept alive.
         *
         * @tparam T_numElements New dimension of the vector.
         * @return First T_numElements elements of the origin vector
         */
        template<uint32_t T_numElements>
        constexpr Vec<T_Type, T_numElements> rshrink() const
        {
            static_assert(T_numElements <= T_dim);
            Vec<T_Type, T_numElements> result{};
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[T_dim - 1u - i];

            return result;
        }

        /** Shrink the vector
         *
         * Removes the last value.
         */
        constexpr Vec<T_Type, T_dim - 1u> eraseBack() const requires(T_dim > 1u)
        {
            constexpr auto reducedDim = T_dim - 1u;
            Vec<T_Type, reducedDim> result{};
            for(uint32_t i = 0u; i < reducedDim; i++)
                result[i] = (*this)[i];

            return result;
        }

        /** Shrink the number of elements of a vector.
         *
         * @tparam T_numElements New dimension of the vector.
         * @param startIdx Index within the origin vector which will be the last element in the result.
         * @return T_numElements elements of the origin vector starting with the index startIdx.
         *         Indexing will wrapp around when the begin of the origin vector is reached.
         */
        template<uint32_t T_numElements>
        constexpr Vec<type, T_numElements> rshrink(int const startIdx) const
        {
            static_assert(T_numElements <= T_dim);
            Vec<type, T_numElements> result;
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[(T_dim + startIdx - i) % T_dim];
            return result;
        }

        /** Removes a component
         *
         * It is not allowed to call this method on a vector with the dimensionality of one.
         *
         * @tparam dimToRemove index which shall be removed; range: [ 0; T_dim - 1 ]
         * @return vector with `T_dim - 1` elements
         */
        template<uint32_t dimToRemove, uint32_t T_deferDim = T_dim, std::enable_if_t<T_deferDim >= 2u, int> = 0>
        constexpr Vec<type, T_dim - 1u> remove() const
        {
            Vec<type, T_dim - 1u> result{};
            for(int i = 0u; i < static_cast<int>(T_dim - 1u); ++i)
            {
                // skip component which must be deleted
                int const sourceIdx = i >= static_cast<int>(dimToRemove) ? i + 1 : i;
                result[i] = (*this)[sourceIdx];
            }
            return result;
        }

        /** Returns product of all components.
         *
         * @return product of components
         */
        constexpr type product() const
        {
            type result = (*this)[0];
            for(uint32_t i = 1u; i < T_dim; i++)
                result *= (*this)[i];
            return result;
        }

        /** Returns sum of all components.
         *
         * @return sum of components
         */
        constexpr type sum() const
        {
            type result = (*this)[0];
            for(uint32_t i = 1u; i < T_dim; i++)
                result += (*this)[i];
            return result;
        }

        /**
         * == comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Vec to compare to
         * @return true if all components in both vectors are equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator==(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            bool result = true;
            for(uint32_t i = 0u; i < T_dim; i++)
                result = result && ((*this)[i] == rhs[i]);
            return result;
        }

        /**
         * != comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Vec to compare to
         * @return true if one component in both vectors are not equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator!=(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            return !((*this) == rhs);
        }

        template<typename T_OtherStorage>
        constexpr auto min(Vec<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            Vec result{};
            for(uint32_t d = 0u; d < T_dim; d++)
                result[d] = std::min((*this)[d], rhs[d]);
            return result;
        }

        /** create string out of the vector
         *
         * @param separator string to separate components of the vector
         * @param enclosings string with dim 2 to enclose vector
         *                   dim == 0 ? no enclose symbols
         *                   dim == 1 ? means enclose symbol begin and end are equal
         *                   dim >= 2 ? letter[0] = begin enclose symbol
         *                               letter[1] = end enclose symbol
         *
         * example:
         * .toString(";","|")     -> |x;...;z|
         * .toString(",","[]")    -> [x,...,z]
         */
        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string locale_enclosing_begin;
            std::string locale_enclosing_end;
            size_t enclosing_dim = enclosings.size();

            if(enclosing_dim > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosing_dim];
                locale_enclosing_end = enclosings[1 % enclosing_dim];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin << (*this)[0];

            for(uint32_t i = 1u; i < T_dim; ++i)
                stream << separator << (*this)[i];
            stream << locale_enclosing_end;
            return stream.str();
        }

        /** swizzle operator */
        template<typename T, T... T_values>
        constexpr auto operator[](Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v) const
        {
            using InType = ALPAKA_TYPEOF(v);
            return Vec<T_Type, InType::dim()>{(*this)[T_values]...};
        }

        template<typename T, T... T_values>
        constexpr auto ref(Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v)
        {
            using InType = ALPAKA_TYPEOF(v);
            auto array = std::array{std::ref((*this)[T_values])...};
            return Vec<T_Type, InType::dim(), ALPAKA_TYPEOF(array)>{array};
        };

        template<typename T, T... T_values>
        constexpr auto ref(Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>> const v) const
        {
            using InType = ALPAKA_TYPEOF(v);
            auto array = std::array{std::ref((*this)[T_values])...};
            return Vec<T_Type, InType::dim(), ALPAKA_TYPEOF(array)>{array};
        };
    };

    template<typename T, T... T_values>
    using CVec = Vec<T, sizeof...(T_values), detail::CVec<T, T_values...>>;

    template<typename T, size_t... T_idx>
    constexpr auto iotaCVecDo(std::index_sequence<T_idx...>)
    {
        return CVec<T, T{T_idx}...>{};
    }

    template<typename T, uint32_t T_dim>
    constexpr auto iotaCVec()
    {
        using IotaSeq = std::make_integer_sequence<size_t, T_dim>;
        return iotaCVecDo<T>(IotaSeq{});
    }

    template<typename AllowedIntegers>
    struct contains
    {
    };

    template<typename Int, Int... Is>
    struct contains<CVec<Int, Is...>>
    {
        template<Int X>
    static constexpr bool value = ((X == Is) || ...);
    };

    template<typename T1, typename T2>
    struct combine
    {
    };

    template<typename Int, Int... Is1, Int... Is2>
    struct combine<CVec<Int, Is1...>, CVec<Int, Is2...>>
    {
        using type = CVec<Int, Is1..., Is2...>;
    };

    template<typename... Seqs>
    struct concatenate;

    template<typename First, typename... Rest>
    struct concatenate<First, Rest...>
    {
        using type = typename combine<First, typename concatenate<Rest...>::type>::type;
    };

    template<typename Last>
    struct concatenate<Last>
    {
        using type = Last;
    };


    template<typename T_V1, typename T_V2>
    struct FilterValues;

    template<typename T1, T1... T_values1, T1... T_values2>
    struct FilterValues<CVec<T1, T_values1...>, CVec<T1, T_values2...>>
    {
        using m0 = CVec<T1, T_values1...>;

        template<T1 N>
        using select = std::conditional_t<contains<m0>::template value<N>, CVec<T1>, CVec<T1, N>>;


        using type = concatenate<select<T_values2>...>::type;
    };

    consteval auto getNotSelected(auto large, auto small)
    {
        return typename FilterValues<ALPAKA_TYPEOF(small), ALPAKA_TYPEOF(large)>::type{};
    }

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    constexpr auto get(Vec<T_Type, T_dim, T_Storage> const& v)
    {
        return v[I];
    }

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    constexpr auto& get(Vec<T_Type, T_dim, T_Storage>& v)
    {
        return v[I];
    }

    template<typename Type>
    struct Vec<Type, 0>
    {
        using type = Type;
        static constexpr uint32_t T_dim = 0;

        template<typename OtherType>
        constexpr operator Vec<OtherType, 0>() const
        {
            return Vec<OtherType, 0>();
        }

        /**
         * == comparison operator.
         *
         * Returns always true
         */
        constexpr bool operator==(Vec const& rhs) const
        {
            return true;
        }

        /**
         * != comparison operator.
         *
         * Returns always false
         */
        constexpr bool operator!=(Vec const& rhs) const
        {
            return false;
        }

        static constexpr Vec create(Type)
        {
            /* this method should never be actually called,
             * it exists only for Visual Studio to handle alpaka::Size_t< 0 >
             */
            static_assert(sizeof(Type) != 0 && false);
        }
    };

    // type deduction guide
    template<typename T_1, typename... T_Args>
    ALPAKA_FN_HOST_ACC Vec(T_1, T_Args...)
        -> Vec<T_1, uint32_t(sizeof...(T_Args) + 1u), ArrayStorage<T_1, uint32_t(sizeof...(T_Args) + 1u)>>;

    template<typename Type, uint32_t T_dim, typename T_Storage>
    std::ostream& operator<<(std::ostream& s, Vec<Type, T_dim, T_Storage> const& vec)
    {
        return s << vec.toString();
    }

/** binary operators
 * @{
 */
#define ALPAKA_VECTOR_BINARY_OP(resultScalarType, op)                                                                 \
    template<typename T_Type, uint32_t T_dim, typename T_Storage, typename T_OtherStorage>                            \
    constexpr auto operator op(                                                                                       \
        const Vec<T_Type, T_dim, T_Storage>& lhs,                                                                     \
        const Vec<T_Type, T_dim, T_OtherStorage>& rhs)                                                                \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs[i];                                                                             \
        return result;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    template<typename T_Type, uint32_t T_dim, typename T_Storage>                                                     \
    constexpr auto operator op(                                                                                       \
        const Vec<T_Type, T_dim, T_Storage>& lhs,                                                                     \
        typename Vec<T_Type, T_dim, T_Storage>::ParamType rhs)                                                        \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs;                                                                                \
        return result;                                                                                                \
    }                                                                                                                 \
    template<typename T_Type, uint32_t T_dim, typename T_Storage>                                                     \
    constexpr auto operator op(                                                                                       \
        typename Vec<T_Type, T_dim, T_Storage>::ParamType lhs,                                                        \
        const Vec<T_Type, T_dim, T_Storage>& rhs)                                                                     \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Vec<resultScalarType, T_dim> result{};                                                                        \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs op rhs[i];                                                                                \
        return result;                                                                                                \
    }
    ALPAKA_VECTOR_BINARY_OP(T_Type, +)
    ALPAKA_VECTOR_BINARY_OP(T_Type, -)
    ALPAKA_VECTOR_BINARY_OP(T_Type, *)
    ALPAKA_VECTOR_BINARY_OP(T_Type, /)
    ALPAKA_VECTOR_BINARY_OP(bool, >=)
    ALPAKA_VECTOR_BINARY_OP(bool, >)
    ALPAKA_VECTOR_BINARY_OP(bool, <=)
    ALPAKA_VECTOR_BINARY_OP(bool, <)
    ALPAKA_VECTOR_BINARY_OP(T_Type, %)

#undef ALPAKA_VECTOR_BINARY_OP

    /** @} */


    /** Give the linear index of an N-dimensional index within an N-dimensional index space.
     *
     * @tparam T_IntegralType vector data type (must be an integral type)
     * @tparam T_dim dimension of the vector, should be >= 2
     * @param dim N-dimensional dim of the index space (N can be one dimension less compared to idx)
     * @param idx N-dimensional index within the index space
     *            @attention behaviour is undefined for negative index
     *            @attention if idx is outside of dim the result will be outside of the the index domain too
     * @return linear index within the index domain
     *
     * @{
     */
    template<
        typename T_IntegralType,
        typename T_Storage,
        typename T_OtherStorage,
        uint32_t T_dim,
        typename = std::enable_if_t<std::is_integral_v<T_IntegralType> && T_dim >= 2>>
    constexpr T_IntegralType linearize(
        Vec<T_IntegralType, T_dim - 1u, T_Storage> const& dim,
        Vec<T_IntegralType, T_dim, T_OtherStorage> const& idx)
    {
        T_IntegralType linearIdx{idx[0]};
        for(uint32_t d = 1u; d < T_dim; ++d)
            linearIdx = linearIdx * dim[d - 1u] + idx[d];

        return linearIdx;
    }

    template<
        typename T_IntegralType,
        typename T_Storage,
        typename T_OtherStorage,
        uint32_t T_dim,
        typename = std::enable_if_t<std::is_integral_v<T_IntegralType>>>
    constexpr T_IntegralType linearize(
        Vec<T_IntegralType, T_dim, T_Storage> const& dim,
        Vec<T_IntegralType, T_dim, T_OtherStorage> const& idx)
    {
        return linearize(dim.template rshrink<T_dim - 1u>(), idx);
    }

    template<
        typename T_IntegralType,
        typename T_Storage,
        typename T_OtherStorage,
        typename = std::enable_if_t<std::is_integral_v<T_IntegralType>>>
    ALPAKA_FN_HOST_ACC T_IntegralType
    linearize(Vec<T_IntegralType, 1u, T_Storage> const&, Vec<T_IntegralType, 1u, T_OtherStorage> const& idx)
    {
        return idx.x();
    }

    /** @} */

    /** Maps a linear index to an N-dimensional index
     *
     * @tparam T_IntegralType vector data type (must be an integral type)
     * @param dim N-dimensional index space
     * @param linearIdx Linear index within dim.
     *        @attention If linearIdx is an index outside of dim the result will be outside of the index domain
     * too.
     * @return N-dimensional index
     *
     * @{
     */
    template<
        typename T_IntegralType,
        typename T_Storage,
        uint32_t T_dim,
        typename = std::enable_if_t<std::is_integral_v<T_IntegralType> && T_dim >= 2u>>
    constexpr auto mapToND(Vec<T_IntegralType, T_dim, T_Storage> const& dim, T_IntegralType linearIdx)
    {
        constexpr uint32_t reducedDim = T_dim - 1u;
        Vec<T_IntegralType, reducedDim> pitchExtents;
        pitchExtents.back() = dim.back();
        for(uint32_t d = 1u; d < T_dim - 1u; ++d)
            pitchExtents[reducedDim - 1u - d] = dim[T_dim - 1u - d] * pitchExtents[reducedDim - d];

        Vec<T_IntegralType, T_dim> result;
        for(uint32_t d = 0u; d < T_dim - 1u; ++d)
        {
            result[d] = linearIdx / pitchExtents[d];
            linearIdx -= pitchExtents[d] * result[d];
        }
        result[T_dim - 1u] = linearIdx;
        return result;
    }

    template<
        typename T_IntegralType,
        typename T_Storage,
        typename = std::enable_if_t<std::is_integral_v<T_IntegralType>>>
    ALPAKA_FN_HOST_ACC auto mapToND(Vec<T_IntegralType, 1u, T_Storage> const& dim, T_IntegralType linearIdx)
    {
        return linearIdx;
    }

    /** @} */


    template<typename T>
    struct IsVector : std::false_type
    {
    };

    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    struct IsVector<Vec<T_Type, T_dim, T_Storage>> : std::true_type
    {
    };

    template<typename T>
    struct IsCVector : std::false_type
    {
    };

    template<typename T_Type, uint32_t T_dim, T_Type... T_values>
    struct IsCVector<Vec<T_Type, T_dim, detail::CVec<T_Type, T_values...>>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isVector_v = IsVector<T>::value;

    template<typename T>
    constexpr bool isCVector_v = IsCVector<T>::value;

    namespace concepts
    {
        template<typename T>
        concept Vector = isVector_v<T>;

        template<typename T>
        concept CVector = isCVector_v<T>;

        template<typename T, typename T_RequiredComponent>
        concept TypeOrVector = (isVector_v<T> || std::is_same_v<T, T_RequiredComponent>);

        template<typename T, typename T_RequiredComponent>
        concept VectorOrConvertableType = (isVector_v<T> || std::is_convertible_v<T, T_RequiredComponent>);
    } // namespace concepts
}; // namespace alpaka

namespace std
{
    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    struct tuple_size<alpaka::Vec<T_Type, T_dim, T_Storage>>
    {
        static constexpr std::size_t value = T_dim;
    };

    template<std::size_t I, typename T_Type, uint32_t T_dim, typename T_Storage>
    struct tuple_element<I, alpaka::Vec<T_Type, T_dim, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
