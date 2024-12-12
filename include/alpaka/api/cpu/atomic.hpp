/* Copyright 2022 Felice Pantaleo, Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/tag.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/atomicHierarchy.hpp"
#include "alpaka/onAcc/atomicOp.hpp"
#include "alpaka/onAcc/internal.hpp"

#include <array>
#include <atomic>
#include <type_traits>

#ifndef ALPAKA_DISABLE_ATOMIC_ATOMICREF
#    ifndef ALPAKA_HAS_STD_ATOMIC_REF
#        include <boost/atomic.hpp>
#    endif

namespace alpaka::onAcc
{
    namespace detail
    {
#    if defined(ALPAKA_HAS_STD_ATOMIC_REF)
        template<typename T>
        using atomic_ref = std::atomic_ref<T>;
#    else
        template<typename T>
        using atomic_ref = boost::atomic_ref<T>;
#    endif
    } // namespace detail

    //! The atomic ops based on atomic_ref for CPU accelerators.
    //
    //  Atomics can be used in the grids, blocks and threads hierarchy levels.
    //

    class AtomicAtomicRef
    {
    };

    template<typename T>
    void isSupportedByAtomicAtomicRef()
    {
        static_assert(
            std::is_trivially_copyable_v<T> && detail::atomic_ref<T>::required_alignment <= alignof(T),
            "Type not supported by AtomicAtomicRef, please recompile defining "
            "ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
    }

    namespace internalCompute
    {
        //! The CPU accelerators AtomicAdd.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicAdd, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_add(value);
            }
        };

        //! The CPU accelerators AtomicSub.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicSub, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_sub(value);
            }
        };

        //! The CPU accelerators AtomicMin.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicMin, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::min(result, value);
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = old;
                    result = std::min(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicMax.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicMax, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = old;
                result = std::max(result, value);
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = old;
                    result = std::max(result, value);
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicExch.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicExch, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = value;
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = value;
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicInc.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicInc, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = ((old >= value) ? 0 : static_cast<T>(old + 1));
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicDec.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicDec, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result = ((old >= value) ? 0 : static_cast<T>(old - 1));
                while(!ref.compare_exchange_weak(old, result))
                {
                    result = ((old >= value) ? 0 : static_cast<T>(old - 1));
                }
                return old;
            }
        };

        //! The CPU accelerators AtomicAnd.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicAnd, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_and(value);
            }
        };

        //! The CPU accelerators AtomicOr.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicOr, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_or(value);
            }
        };

        //! The CPU accelerators AtomicXor.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicXor, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(internal::StlAtomic const&, T* const addr, T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                return ref.fetch_xor(value);
            }
        };

        //! The CPU accelerators AtomicCas.
        template<typename T, typename THierarchy>
        struct Atomic::Op<AtomicCas, internal::StlAtomic, T, THierarchy>
        {
            ALPAKA_FN_HOST static auto atomicOp(
                internal::StlAtomic const&,
                T* const addr,
                T const& compare,
                T const& value) -> T
            {
                isSupportedByAtomicAtomicRef<T>();
                detail::atomic_ref<T> ref(*addr);
                T old = ref;
                T result;
                do
                {
#    if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wfloat-equal"
#    endif
                    result = ((old == compare) ? value : old);
#    if ALPAKA_COMP_GNUC || ALPAKA_COMP_CLANG
#        pragma GCC diagnostic pop
#    endif
                } while(!ref.compare_exchange_weak(old, result));
                return old;
            }
        };
    } // namespace internalCompute
} // namespace alpaka::onAcc

#endif
