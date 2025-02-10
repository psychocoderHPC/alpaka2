/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"

#include <type_traits>

namespace alpaka
{
    template<typename T_Type, typename T_Extents, typename T_Pitches>
    struct MdSpan
    {
        using element_type = T_Type;
        using reference = element_type&;
        using index_type = typename T_Pitches::type;

        static_assert(std::is_convertible_v<index_type, typename T_Extents::type>);

        static consteval uint32_t dim()
        {
            return T_Extents::dim();
        }

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        constexpr reference operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * @{
         */
        constexpr element_type const* data() const
        {
            return this->m_ptr;
        }

        constexpr element_type* data()
        {
            return this->m_ptr;
        }

        /** @} */

        /*Object must init by copy a valid instance*/
        constexpr MdSpan() = default;

        /** Constructor
         *
         * @param pointer pointer to the memory
         * @param extents number of elements
         * @param pitchBytes pitch in bytes per dimension
         */
        constexpr MdSpan(element_type* pointer, T_Extents extents, T_Pitches const& pitchBytes)
            : m_ptr(pointer)
            , m_extent(extents)
            , m_pitch(pitchBytes.eraseBack())
        {
        }

        MdSpan(MdSpan const&) = default;
        MdSpan(MdSpan&&) = default;

        /** get value at the given index
         *
         * @param idx n-dimensional offset, relative to the origin pointer
         * @return reference to the value
         * @{
         */
        constexpr element_type const& operator[](concepts::Vector auto const& idx) const
        {
            return *ptr(idx);
        }

        constexpr reference operator[](concepts::Vector auto const& idx)
        {
            return *const_cast<element_type*>(ptr(idx));
        }

        /** }@ */

        constexpr auto getExtents() const
        {
            return m_extent;
        }

        /** shift the access by idx elements
         *
         * @attention The original extents will be lost and getExtents() is the number of valid elements until the end.
         *
         * @param idx number of elements to jump over
         * @return shifted access with origin points to the idx'ed element of the original memory
         *
         * @{
         */
        constexpr auto shift(concepts::Vector auto const& idx) const
        {
            auto result = *this;
            result.m_ptr = &(*this)[idx];
            result.m_extent = m_extent - idx;

            return result;
        }

        constexpr auto shift(concepts::Vector auto const& idx)
        {
            auto result = *this;
            result.m_ptr = &(*this)[idx];
            result.m_extent = m_extent - idx;

            return result;
        }
        /** @} */

    protected:
        /** get the pointer of the value relative to the origin pointer m_ptr
         *
         * @param idx n-dimensional offset
         * @return pointer to value
         */
        constexpr element_type const* ptr(concepts::Vector auto const& idx) const
        {
            /** offset in bytes
             *
             * We calculate the complete offset in bytes even if it would be possible to change the x-dimension
             * with the native element_types pointer, this is reducing the register footprint.
             */
            index_type offset = sizeof(element_type) * idx.back();
            for(uint32_t d = 0u; d < dim() - 1u; ++d)
            {
                offset += m_pitch[d] * idx[d];
            }
            return reinterpret_cast<element_type const*>(reinterpret_cast<char const*>(this->m_ptr) + offset);
        }

        element_type* m_ptr;
        T_Extents m_extent;
        decltype(std::declval<T_Pitches>().eraseBack()) m_pitch;
    };

    template<typename T_Type, typename T_Extents, typename T_Pitches>
    ALPAKA_FN_HOST_ACC MdSpan(T_Type* pointer, T_Extents const&, T_Pitches const&)
        -> MdSpan<T_Type, T_Extents, T_Pitches>;

    template<typename T_Type, typename T_Extents, typename T_Pitches>
    requires(T_Pitches::dim() == 1u && T_Extents::dim() == 1u)
    struct MdSpan<T_Type, T_Extents, T_Pitches>
    {
        using element_type = T_Type;
        using reference = element_type&;
        using index_type = typename T_Pitches::type;

        static_assert(std::is_convertible_v<index_type, typename T_Extents::type>);

        static consteval uint32_t dim()
        {
            return 1u;
        }

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        constexpr reference operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * @{
         */
        constexpr element_type const* data() const
        {
            return this->m_ptr;
        }

        constexpr element_type* data()
        {
            return this->m_ptr;
        }

        /** @} */

        /*Object must init by copy a valid instance*/
        constexpr MdSpan() = default;

        /** Constructor
         *
         * @param pointer pointer to the memory
         * @param extents number of elements
         * @param pitchBytes pitch in bytes per dimension
         */
        constexpr MdSpan(element_type* pointer, T_Extents const& extents, [[maybe_unused]] T_Pitches const& pitchBytes)
            : m_ptr(pointer)
            , m_extent(extents)
        {
        }

        constexpr MdSpan(element_type* pointer) : m_ptr(pointer)
        {
        }

        constexpr MdSpan(MdSpan const&) = default;
        constexpr MdSpan(MdSpan&&) = default;

        /** get value at the given index
         *
         * @param idx offset relative to the origin pointer
         * @return reference to the value
         * @{
         */
        constexpr element_type const& operator[](concepts::Vector auto const& idx) const
        {
            return *(m_ptr + idx.x());
        }

        constexpr reference operator[](concepts::Vector auto const& idx)
        {
            return *(m_ptr + idx.x());
        }

        constexpr element_type const& operator[](std::integral auto const& idx) const
        {
            return *(m_ptr + idx);
        }

        constexpr reference operator[](std::integral auto const& idx)
        {
            return *(m_ptr + idx);
        }

        constexpr bool operator==(MdSpan const other) const
        {
            return m_ptr == other.m_ptr && m_extent == other.m_extent;
        }

        /** @} */

        constexpr auto getExtents() const
        {
            return m_extent;
        }

        /** shift the access by idx elements
         *
         * @attention The original extents will be lost and getExtents() is the number of valid elements until the end.
         *
         * @param idx number of elements to jump over
         * @return shifted access with origin points to the idx'ed element of the original memory
         *
         * @{
         */
        constexpr auto shift(concepts::Vector auto const& idx) const
        {
            auto result = *this;
            result.m_ptr = &(*this)[idx];
            result.m_extent = m_extent - idx;

            return result;
        }

        constexpr auto shift(concepts::Vector auto const& idx)
        {
            auto result = *this;
            result.m_ptr = &(*this)[idx];
            result.m_extent = m_extent - idx;

            return result;
        }

        /** @} */

    protected:
        element_type* m_ptr;
        T_Extents m_extent;
    };

    /** access a C array with compile time extents via a runtime md index. */
    template<std::integral auto T_numDims, uint32_t T_dim = 0u>
    struct ResolveArrayAccess
    {
        constexpr decltype(auto) operator()(auto arrayPtr, concepts::Vector auto const& idx) const
        {
            return ResolveArrayAccess<T_numDims - 1u, T_dim + 1u>{}(arrayPtr[idx[T_dim]], idx);
        }
    };

    template<uint32_t T_dim>
    struct ResolveArrayAccess<1u, T_dim>
    {
        constexpr decltype(auto) operator()(auto arrayPtr, concepts::Vector auto const& idx) const
        {
            return arrayPtr[idx[T_dim]];
        }
    };

    /** build C array type with compile time extents from a scalar value based on the compile time extents vector */
    template<typename T, concepts::CVector T_Extent, uint32_t T_numDims = T_Extent::dim(), uint32_t T_dim = 0u>
    struct CArrayType
    {
        using type = typename CArrayType<T[T_Extent{}[T_dim]], T_Extent, T_numDims - 1u, T_dim + 1u>::type;
    };

    template<typename T, concepts::CVector T_Extent, uint32_t T_dim>
    struct CArrayType<T, T_Extent, 1u, T_dim>
    {
        using type = T[T_Extent{}[T_dim]];
    };

    namespace detail
    {
        template<typename T_ArrayType>
        requires(std::is_array_v<T_ArrayType>)
        inline constexpr auto getExtents()
        {
            constexpr uint32_t dim = std::rank_v<T_ArrayType>;
            using index_type = uint32_t;

            constexpr auto createExtents = []<size_t... extents>(std::index_sequence<extents...>) constexpr
            { return alpaka::CVec<index_type, std::extent_v<T_ArrayType, static_cast<index_type>(extents)>...>{}; };

            return createExtents(std::make_integer_sequence<size_t, dim>{});
        }

    } // namespace detail

    template<typename T_ArrayType, typename T_Extents = decltype(detail::getExtents<T_ArrayType>())>
    struct MdSpanArray
    {
        static_assert(
            sizeof(T_ArrayType) && false,
            "MdSpanArray can only be used if std::is_array_v<T> is true for the given type.");
    };

    template<typename T_ArrayType, typename T_Extents>
    requires(std::is_array_v<T_ArrayType>)
    struct MdSpanArray<T_ArrayType, T_Extents>
    {
        using extentType = std::extent<T_ArrayType, std::rank_v<T_ArrayType>>;
        using element_type = std::remove_all_extents_t<T_ArrayType>;
        using reference = element_type&;
        using index_type = typename extentType::value_type;

        static consteval uint32_t dim()
        {
            return std::rank_v<T_ArrayType>;
        }

        /** return value the origin pointer is pointing to
         *
         * @return value at the current location
         */
        constexpr reference operator*()
        {
            return *this->m_ptr;
        }

        /** get origin pointer
         *
         * @{
         */
        constexpr element_type const* data() const
        {
            return this->m_ptr;
        }

        constexpr element_type* data()
        {
            return this->m_ptr;
        }

        /** @} */

        /*Object must init by copy a valid instance*/
        constexpr MdSpanArray() = default;

        /** Constructor
         *
         * @param pointer pointer to the memory
         * @param extents number of elements
         * @param pitchBytes pitch in bytes per dimension
         */
        constexpr MdSpanArray(T_ArrayType& staticSizedArray) : m_ptr(staticSizedArray), m_extents(T_Extents{})
        {
        }

        constexpr MdSpanArray(T_ArrayType& staticSizedArray, T_Extents const & extents) : m_ptr(staticSizedArray), m_extents(extents)
        {
        }

        constexpr MdSpanArray(MdSpanArray const&) = default;
        constexpr MdSpanArray(MdSpanArray&&) = default;

        /** get value at the given index
         *
         * @param idx offset relative to the origin pointer
         * @return reference to the value
         * @{
         */
        constexpr element_type const& operator[](concepts::Vector auto const& idx) const
        {
            return ResolveArrayAccess<dim()>{}(m_ptr, idx);
        }

        constexpr reference operator[](concepts::Vector auto const& idx)
        {
            return ResolveArrayAccess<dim()>{}(m_ptr, idx);
        }

        constexpr element_type const& operator[](index_type const& idx) const
        {
            return m_ptr[idx];
        }

        constexpr reference operator[](index_type const& idx)
        {
            return m_ptr[idx];
        }

        constexpr bool operator==(MdSpanArray const other) const
        {
            return m_ptr == other.m_ptr;
        }

        /** @} */

        constexpr auto getExtents() const
        {
            return m_extents;
        }

        /** shift the access by idx elements
         *
         * @attention A shifted object is loosing the compile time extents information.
         * The original extents will be lost and getExtents() is the number of valid elements until the end.
         *
         * @param idx number of elements to jump over
         * @return shifted access with origin points to the idx'ed element of the original memory
         *
         * @{
         */
        constexpr auto shift(concepts::Vector auto const& idx) const
        {
            auto extents = m_extents - idx;
            return MdSpanArray<T_ArrayType, ALPAKA_TYPEOF(extents)>{
                *reinterpret_cast<T_ArrayType*>(&(*this)[idx]),
                extents};
        }

        constexpr auto shift(concepts::Vector auto const& idx)
        {
            auto extents = m_extents - idx;
            return MdSpanArray<T_ArrayType, ALPAKA_TYPEOF(extents)>{
                *reinterpret_cast<T_ArrayType*>(&(*this)[idx]),
                extents};
        }

        /** @} */

    protected:
        T_ArrayType& m_ptr;
        T_Extents m_extents;
    };
} // namespace alpaka
