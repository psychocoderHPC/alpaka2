/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/core/common.hpp"
#if ALPAKA_OMP

#    include "alpaka/api/cpu/block/mem/SharedStorage.hpp"
#    include "alpaka/core/Vectorize.hpp"

namespace alpaka::onAcc
{
    namespace cpu
    {
        template<std::size_t TDataAlignBytes = core::vectorization::defaultAlignment>
        struct OmpStaticShared : private detail::SharedStorage<TDataAlignBytes>
        {
            template<typename T, size_t T_unique>
            T& allocVar()
            {
                using Base = detail::SharedStorage<TDataAlignBytes>;

                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    // Assure that all threads have executed the return of the last allocBlockSharedArr function (if
                    // there was one before).
#    pragma omp barrier
#    pragma omp single nowait
                    {
                        Base::template alloc<T>(T_unique);
                    }

#    pragma omp barrier
                    // lookup for the data chunk allocated by the master thread
                    data = Base::template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }

            template<typename T, size_t T_unique>
            T* allocDynamic(uint32_t numBytes)
            {
                using Base = detail::SharedStorage<TDataAlignBytes>;

                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    // Assure that all threads have executed the return of the last allocBlockSharedArr function (if
                    // there was one before).
#    pragma omp barrier
#    pragma omp single nowait
                    {
                        Base::template allocDynamic<T>(T_unique, numBytes);
                    }

#    pragma omp barrier
                    // lookup for the data chunk allocated by the master thread
                    data = Base::template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return data;
            }

            void reset()
            {
            }
        };
    } // namespace cpu
} // namespace alpaka::onAcc

#endif
