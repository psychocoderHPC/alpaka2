/* Copyright 2026 SiPearl
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

/** Enable parallel memcpy for CPU backend at compile time.
 *
 * Default: disabled (sequential memcpy)
 * Can be overridden at runtime via ALPAKA_MEMCPY_MODE environment variable or
 * fine tuned in the code using available functions
 *
 */

#ifndef ALPAKA_PARALLEL_MEMCPY_HOST_AS_DEFAULT
#    define ALPAKA_PARALLEL_MEMCPY_HOST_AS_DEFAULT false
#endif

namespace alpaka::onHost::config
{
    /** Memcpy execution mode enumeration
     *
     * Controls whether memcpy operations use sequential (std::memcpy) or
     * parallel (kernel-based) implementation.
     */
    enum class MemcpyMode
    {
        sequential, /**< Use standard std::memcpy */
        parallel /**< Use kernel-based parallelization across threads */
    };

    /** Per-queue memcpy configuration
     *
     * Each CPU queue has its own independent memcpy configuration instance.
     * Configuration is initialized from environment variables when the queue is created.
     * Can be modified at runtime for per-queue customization without affecting other queues.
     *
     * Environment variables (read at queue creation time):
     * - ALPAKA_MEMCPY_MODE: "sequential" or "parallel" (default: compile-time flag)
     * - ALPAKA_MEMCPY_NUM_CORES: number of cores to use (0 = all available cores)
     * - ALPAKA_MEMCPY_MIN_SIZE: minimum copy size in bytes to trigger parallel mode
     *
     */
    class MemcpyConfig
    {
    public:
        /** Initialize queue memcpy configuration from environment variables
         *
         * Call reset() that:
         * - Restore to Alpaka3 default values (defined in reset() )
         * - Reads environment variables at construction time.
         *
         */
        MemcpyConfig()
        {
            reset();
        }

        /** Get current memcpy mode
         *
         * @return Current MemcpyMode (sequential or parallel)
         */
        MemcpyMode getMode() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_mode;
        }

        /** Set memcpy mode
         *
         * Can be changed at any time during program execution.
         * Affects all subsequent memcpy operations on this queue.
         *
         * @param mode The new MemcpyMode to use (sequential or parallel)
         */
        void setMode(MemcpyMode mode)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_mode = mode;
        }

        // FIXME : 'numCores' is a chunk count, the thread pool of the backend decides the real concurrency
        /** Get number of CPU cores to use for parallel memcpy
         *
         * @return Number of cores to use
         *   - 0: Use all available CPU cores
         *   - N > 0: Use up to N cores, capped to the number of available cores
         */
        uint32_t getNumCores() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_numCores;
        }

        /** Set number of CPU cores to use for parallel memcpy
         *
         * The value is the number of chunks the copy is split into and is capped to the number of cores of the numa
         * domain of the queue. The number of cores used is an indirect consequence of it, the chunks are enqueued
         * as thread blocks and distributed over the thread pool of the backend, this value does not size that pool.
         *
         * Can be changed at any time during program execution.
         *
         * @param numCores Number of cores to use
         *   - 0: Use all available CPU cores (default)
         *   - N > 0: Limit to N cores (useful for load balancing)
         */
        void setNumCores(uint32_t numCores)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_numCores = numCores;
        }

        /** Get the minimum copy size (in bytes) to use parallel memcpy
         *
         * @return Minimum size in bytes for triggering parallel memcpy
         */
        size_t getMinSizeForParallel() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_minSizeForParallel;
        }

        /** Set the minimum copy size (in bytes) to use parallel memcpy
         *
         * Copies smaller than this threshold will always use sequential std::memcpy
         * even if mode is set to parallel. To avoid parallelization overhead for small copies.
         *
         * Can be changed at any time during program execution.
         *
         * @param minSize Minimum size in bytes
         *   - 0: No minimum threshold, parallelize all copies
         *   - 1 MB (default): Only parallelize copies >= 1 MB
         *   - N bytes: Only parallelize copies >= N bytes
         */
        void setMinSizeForParallel(size_t minSize)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_minSizeForParallel = minSize;
        }

        /** Check if parallel memcpy should be used for given size
         *
         * This function combines mode and size threshold to determine
         * whether parallel memcpy should be used.
         *
         * @param sizeInBytes Total number of bytes to copy
         * @return true if parallel memcpy should be used, false otherwise
         */
        bool shouldUseParallel(size_t sizeInBytes) const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_mode == MemcpyMode::parallel && sizeInBytes >= m_minSizeForParallel;
        }

        /** Reset configuration to defaults from environment or compile-time defaults
         *
         * This function resets to compile-time defaults and then re-parses environment variables.
         * Useful for resetting configuration after runtime modifications.
         *
         * If environment variables are set, they override the compile-time defaults.
         * Else uses compile-time defaults:
         * - mode: value of ALPAKA_PARALLEL_MEMCPY_HOST
         * - numCores: 0 (all available)
         * - minSizeForParallel: 1 MB
         */
        void reset()
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            // Reset to compile-time defaults first
            m_mode = ALPAKA_PARALLEL_MEMCPY_HOST_AS_DEFAULT ? MemcpyMode::parallel : MemcpyMode::sequential;
            m_numCores = 0;
            m_minSizeForParallel = 1024 * 1024; // 1 MB default

            // Parse ALPAKA_MEMCPY_MODE
            // Values: "sequential", "serial", or "parallel"
            char const* modeEnv = std::getenv("ALPAKA_MEMCPY_MODE");
            if(modeEnv != nullptr)
            {
                std::string modeStr(modeEnv);
                // Convert to lowercase for case-insensitive comparison
                for(auto& c : modeStr)
                    c = std::tolower(static_cast<unsigned char>(c));

                if(modeStr == "sequential" || modeStr == "serial")
                {
                    m_mode = MemcpyMode::sequential;
                }
                else if(modeStr == "parallel")
                {
                    m_mode = MemcpyMode::parallel;
                }
                else
                {
                    throw std::invalid_argument(
                        "ALPAKA_MEMCPY_MODE: invalid value '" + modeStr + "', must be 'sequential' or 'parallel'");
                }
            }

            // Parse ALPAKA_MEMCPY_NUM_CORES
            // Value: non-negative integer (0 = all cores)
            char const* numCoresEnv = std::getenv("ALPAKA_MEMCPY_NUM_CORES");
            if(numCoresEnv != nullptr)
            {
                try
                {
                    int cores = std::stoi(numCoresEnv);
                    if(cores < 0)
                        throw std::invalid_argument("must be >= 0");
                    m_numCores = static_cast<uint32_t>(cores);
                }
                catch(std::exception const& e)
                {
                    throw std::invalid_argument(std::string("ALPAKA_MEMCPY_NUM_CORES: ") + e.what());
                }
            }

            // Parse ALPAKA_MEMCPY_MIN_SIZE
            // Value: non-negative integer representing bytes
            char const* minSizeEnv = std::getenv("ALPAKA_MEMCPY_MIN_SIZE");
            if(minSizeEnv != nullptr)
            {
                try
                {
                    long long minSize = std::stoll(minSizeEnv);
                    if(minSize < 0)
                        throw std::invalid_argument("must be >= 0");
                    m_minSizeForParallel = static_cast<size_t>(minSize);
                }
                catch(std::exception const& e)
                {
                    throw std::invalid_argument(std::string("ALPAKA_MEMCPY_MIN_SIZE: ") + e.what());
                }
            }
        }

        /** Get current configuration as a human-readable string (thread-safe)
         *
         * Useful for debugging, logging, and displaying current settings.
         *
         * @return String representation of current configuration
         * Example: "MemcpyConfig{mode=parallel, numCores=4, minSizeForParallel=2097152 bytes}"
         */
        std::string getConfigString() const
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::string result = "MemcpyConfig{mode=";
            result += (m_mode == MemcpyMode::sequential) ? "sequential" : "parallel";
            result += ", numCores=" + std::to_string(m_numCores == 0 ? 0 : m_numCores);
            result += ", minSizeForParallel=" + std::to_string(m_minSizeForParallel) + " bytes}";
            return result;
        }

    private:
        /** Mutex protecting configuration state */
        mutable std::mutex m_mutex;

        /** Current memcpy execution mode */
        MemcpyMode m_mode;

        /** Number of CPU cores to use for parallel memcpy
         * 0 means use all available cores
         */
        size_t m_numCores;

        /** Minimum copy size in bytes to trigger parallel memcpy
         * Copies smaller than this will always use sequential memcpy
         */
        size_t m_minSizeForParallel;
    };

} // namespace alpaka::onHost::config
