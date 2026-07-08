/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/hwloc/hwlocConfig.hpp"
#include "alpaka/api/host/sysInfo.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/unused.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

/** Implement functions required to set thread affinity and pin memory.
 *
 * There is always a fallback implementation to be able to run without hwloc.
 * In this case domain selection is not possible and all cores will be taken into account.
 *
 * Domain model:
 *  - A CPU domain is the object used as an alpaka host device.
 *  - If hwloc Group objects with CPU sets are available, CPU domains are Groups.
 *  - Otherwise, CPU domains fall back to NUMA nodes with CPU sets.
 *  - NUMA nodes are memory targets attached to a CPU domain, not necessarily CPU devices themselves.
 *
 * This avoids duplicating the same CPU cores as multiple alpaka devices on systems where DDR and HBM are modeled as
 * separate NUMA nodes below the same Group.
 */
namespace alpaka::onHost::internal::hwloc
{
    /** Constant to select all CPU domains. */
    constexpr uint32_t allDomains = std::numeric_limits<uint32_t>::max();

#if ALPAKA_HAS_HWLOC
    /** Helper singleton to cache the hwloc topology.
     *
     * Caching is required to reduce the overhead for repeating operations.
     * Building the topology can be expensive.
     */
    class TopologyCache
    {
    public:
        static TopologyCache& instance()
        {
            static TopologyCache topology;
            return topology;
        }

        hwloc_topology_t get() const noexcept
        {
            return m_topology;
        }

    private:
        TopologyCache()
        {
            if(hwloc_topology_init(&m_topology) != 0)
            {
                throw std::runtime_error("hwloc_topology_init failed");
            }
            if(hwloc_topology_load(m_topology) != 0)
            {
                hwloc_topology_destroy(m_topology);
                throw std::runtime_error("hwloc_topology_load failed");
            }
        }

        ~TopologyCache()
        {
            if(m_topology != nullptr)
            {
                hwloc_topology_destroy(m_topology);
            }
        }

        TopologyCache(TopologyCache const&) = delete;
        TopologyCache& operator=(TopologyCache const&) = delete;
        TopologyCache(TopologyCache&&) = delete;
        TopologyCache& operator=(TopologyCache&&) = delete;

    private:
        hwloc_topology_t m_topology{};
    };

    [[noreturn]] inline void throwErrno(char const* what)
    {
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
    }

    /** Shorthand to get the cached hwloc topology cache */
    inline hwloc_topology_t getTopology()
    {
        return TopologyCache::instance().get();
    }

    /** Check if there are CPUs under the object
     *
     * It is possible to create numa domains which does not contain CPUs, those we do not want to use.
     *
     * @return true if the object has CPUs, else false.
     */
    inline bool hasNonEmptyCpuSet(hwloc_obj_t obj)
    {
        return obj != nullptr && obj->cpuset != nullptr && !hwloc_bitmap_iszero(obj->cpuset);
    }

    /** Check if a hwloc group exists
     *
     * With a hwloc group it is possible to combine more than one numa domain together with cores.
     *
     * @return true if 'subtreeRoot' contains a group, else false
     */
    inline bool isObjectInSubtree(hwloc_obj_t obj, hwloc_obj_t subtreeRoot)
    {
        for(hwloc_obj_t current = obj; current != nullptr; current = current->parent)
        {
            if(current == subtreeRoot)
                return true;
        }
        return false;
    }

    /** Search for all hwloc groups.
     *
     * @return A list with hwloc objects where each object is a group. Each group guarantees to contain numa domains
     * and CPUs.
     */
    inline std::vector<hwloc_obj_t> getGroupCpuDomains()
    {
        std::vector<hwloc_obj_t> domains;
        hwloc_topology_t const topology = getTopology();

        hwloc_obj_t group = nullptr;
        while((group = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_GROUP, group)) != nullptr)
        {
            // skip if no CPUs are present in the group
            if(!hasNonEmptyCpuSet(group))
                continue;

            // Use only groups that actually contain a NUMA node. This avoids selecting artificial grouping levels that
            // do not describe a CPU/memory locality domain.
            bool containsNumaNode = false;
            hwloc_obj_t node = nullptr;
            while((node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) != nullptr)
            {
                if(isObjectInSubtree(node, group))
                {
                    containsNumaNode = true;
                    break;
                }
            }

            if(containsNumaNode)
                domains.push_back(group);
        }

        return domains;
    }

    /** Search for all hwloc numa domains.
     *
     * @return A list with hwloc objects where each object is a numa domain. Each domain guarantees to contain CPUs.
     */
    inline std::vector<hwloc_obj_t> getNumaCpuDomains()
    {
        std::vector<hwloc_obj_t> domains;
        hwloc_topology_t const topology = getTopology();

        hwloc_obj_t node = nullptr;
        while((node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) != nullptr)
        {
            if(hasNonEmptyCpuSet(node))
                domains.push_back(node);
        }

        return domains;
    }

    /** Return the host CPU domains
     *
     @return List with hwloc Groups with CPU sets and NUMA children, otherwise fall back to NUMA nodes with CPU sets.
     */
    inline std::vector<hwloc_obj_t> getCpuDomains()
    {
        std::vector<hwloc_obj_t> domains = getGroupCpuDomains();
        if(!domains.empty())
            return domains;

        return getNumaCpuDomains();
    }

    /** Return the hwloc object for the domain index. */
    inline hwloc_obj_t getCpuDomainObj(uint32_t domainIdx)
    {
        std::vector<hwloc_obj_t> const domains = getCpuDomains();
        if(domainIdx >= domains.size())
        {
            throw std::out_of_range("CPU domain index out of range: " + std::to_string(domainIdx));
        }
        return domains[domainIdx];
    }

    /** Return all hwloc numa domains of an object.
     *
     * @return List of all numa domains under an object. It is guaranteed that each object has at least one numa
     * domain.
     */
    inline std::vector<hwloc_obj_t> getMemoryNodes(hwloc_obj_t domainObj)
    {
        std::vector<hwloc_obj_t> nodes;
        hwloc_topology_t const topology = getTopology();

        if(domainObj == nullptr)
            return nodes;

        if(domainObj->type == HWLOC_OBJ_NUMANODE)
        {
            nodes.push_back(domainObj);
            return nodes;
        }

        hwloc_obj_t node = nullptr;
        while((node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) != nullptr)
        {
            if(isObjectInSubtree(node, domainObj))
                nodes.push_back(node);
        }

        return nodes;
    }

#endif

    /** Get the number of alpaka host CPU domains. */
    inline uint32_t getNumCpuDomains()
    {
#if ALPAKA_HAS_HWLOC
        std::vector<hwloc_obj_t> const domains = getCpuDomains();
        return static_cast<uint32_t>(domains.size());
#else
        return 1u;
#endif
    }

    /** Parse the OS NUMA information.
     *
     * hwloc is not providing the available free memory in a NUMA domain.
     * Therefore we fall back to check the NUMA node information in the OS directly.
     *
     * @param osNodeIndex The index of the NUMA node in the OS.
     * @param key The key value you want to read out e.g. 'MemFree:' or 'HugePages_Total:'.
     */
    inline std::optional<size_t> parseNodeMemInfoValueBytes(unsigned osNodeIndex, std::string_view key)
    {
        std::ifstream in("/sys/devices/system/node/node" + std::to_string(osNodeIndex) + "/meminfo");
        if(!in)
        {
            return std::nullopt;
        }

        std::string line;
        while(std::getline(in, line))
        {
            if(line.find(std::string(key)) == std::string::npos)
            {
                continue;
            }

            // Example line:
            // Node 0 MemFree:        123456 kB
            std::istringstream iss(line);
            std::string nodeWord;
            unsigned nodeNumber = 0;
            std::string field;
            size_t valueKB = 0;
            std::string unit;
            if(iss >> nodeWord >> nodeNumber >> field >> valueKB >> unit)
            {
                if(field == key && unit == "kB")
                {
                    return valueKB * 1024ULL;
                }
            }
        }

        return std::nullopt;
    }

    /** Set the affinity of the current thread to all cores of a CPU domain.
     *
     * @param cpuDomainIdx Legacy name: CPU-domain index starting with zero, or 'allDomains' to use all cores.
     */
    inline void setThreadAffinity(uint32_t cpuDomainIdx)
    {
#if ALPAKA_HAS_HWLOC
        hwloc_cpuset_t cpuset = nullptr;

        if(cpuDomainIdx == allDomains)
        {
            hwloc_const_cpuset_t const fullSet = hwloc_topology_get_complete_cpuset(getTopology());
            if(fullSet == nullptr)
            {
                throw std::runtime_error("Topology has no complete cpuset");
            }

            cpuset = hwloc_bitmap_dup(fullSet);
        }
        else
        {
            hwloc_obj_t const domain = getCpuDomainObj(cpuDomainIdx);
            if(domain->cpuset == nullptr || hwloc_bitmap_iszero(domain->cpuset))
            {
                throw std::runtime_error("CPU domain has no cpuset");
            }

            cpuset = hwloc_bitmap_dup(domain->cpuset);
        }

        if(cpuset == nullptr)
        {
            throw std::bad_alloc();
        }

        int const rc = hwloc_set_cpubind(getTopology(), cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);

        hwloc_bitmap_free(cpuset);

        if(rc != 0)
        {
            throwErrno("hwloc_set_cpubind failed");
        }
#else
        alpaka::unused(cpuDomainIdx);
        return;
#endif
    }

#if ALPAKA_HAS_HWLOC
    /** Set the NUMA memory target for the memory range described by ptr and bytes. */
    template<typename T>
    inline void pinPointerToNumaNode(T* const ptr, size_t bytes, hwloc_obj_t const node)
    {
        if(ptr == nullptr || bytes == 0u)
            return;

        if(node == nullptr)
            throw std::runtime_error("NUMA node is null");

        if(node->type != HWLOC_OBJ_NUMANODE)
            throw std::runtime_error("Memory binding target is not a NUMA node");

        if(node->nodeset == nullptr)
        {
            throw std::runtime_error("NUMA node has no nodeset");
        }

        hwloc_nodeset_t nodeset = hwloc_bitmap_dup(node->nodeset);
        if(nodeset == nullptr)
        {
            throw std::bad_alloc();
        }

        int const rc = hwloc_set_area_membind(
            getTopology(),
            alpaka::toVoidPtr(ptr),
            bytes,
            nodeset,
            HWLOC_MEMBIND_BIND,
            HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT);

        hwloc_bitmap_free(nodeset);

        if(rc != 0)
        {
#    ifdef ALPAKA_HOST_MEM_PINNING_CAN_FAIL
            // Missing privileges, e.g. within a container.
            bool const operationNotSupported = errno == EPERM;
            // Unsupported platform.
            bool const functionNotImplemented = errno == ENOSYS;
            // NUMA node is not allowed by cpuset/cgroup.
            bool const operationNotAllowed = errno == EXDEV;
            if(operationNotSupported || functionNotImplemented || operationNotAllowed)
            {
                return;
            }
#    endif
            throwErrno("hwloc_set_area_membind failed");
        }
    }
#endif

    /** Set the default NUMA memory node for a CPU-domain memory range.
     *
     * @attention This method should be called before the memory is touched, else it may have no effect.
     *            If a cpuDomainIdx contains more than one numa domain, the first numa domain will be used.
     *
     * @param ptr pointer address to pin, nullptr is valid input.
     * @param bytes the number of bytes to pin starting from the ptr address.
     * @param cpuDomainIdx Index of the cpu group.
     */
    template<typename T>
    inline void pinPointer(T* const ptr, size_t bytes, uint32_t cpuDomainIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(cpuDomainIdx == allDomains)
            return;

        if(ptr == nullptr || bytes == 0u)
            return;

        std::vector<hwloc_obj_t> const nodes = getMemoryNodes(getCpuDomainObj(cpuDomainIdx));
        if(nodes.empty())
        {
            throw std::runtime_error("CPU domain has no associated NUMA memory node");
        }

        /** take the first numa domain
         *
         * @todo: this should be slectable during onHost::alloc()
         */
        pinPointerToNumaNode(ptr, bytes, nodes.front());
#else
        alpaka::unused(ptr, bytes, cpuDomainIdx);
        return;
#endif
    }

    /** Return the number of cores in a CPU domain.
     *
     * Here "cores" means logical CPUs / processing units, so SMT siblings are counted too.
     *
     * @param cpuDomainIdx Index of the cpu group.
     */
    inline uint32_t getNumCores(uint32_t cpuDomainIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(cpuDomainIdx == allDomains)
            return std::thread::hardware_concurrency();

        hwloc_obj_t const domain = getCpuDomainObj(cpuDomainIdx);
        if(domain->cpuset == nullptr || hwloc_bitmap_iszero(domain->cpuset))
        {
            throw std::runtime_error("CPU domain has no cpuset");
        }

        int const numPUs = hwloc_bitmap_weight(domain->cpuset);
        if(numPUs < 0)
        {
            throw std::runtime_error("hwloc_bitmap_weight failed");
        }

        return static_cast<uint32_t>(numPUs);
#else
        alpaka::unused(cpuDomainIdx);
        return std::thread::hardware_concurrency();
#endif
    }

    /** Return the memory capacity of a CPU domain.
     *
     * If the CPU domain has multiple attached NUMA memory nodes, e.g. DDR and HBM, the returned value is the sum of
     * all attached memory nodes.
     *
     * @param cpuDomainIdx Index of the cpu group.
     */
    inline size_t getMemCapacityBytes(uint32_t cpuDomainIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(cpuDomainIdx == allDomains)
            return alpaka::onHost::getGlobalMemCapacityBytes();

        size_t bytes = 0u;
        for(hwloc_obj_t const node : getMemoryNodes(getCpuDomainObj(cpuDomainIdx)))
        {
            if(node->attr == nullptr)
            {
                throw std::runtime_error("NUMA node has no attributes");
            }
            bytes += static_cast<size_t>(node->attr->numanode.local_memory);
        }

        return bytes;
#else
        alpaka::unused(cpuDomainIdx);
        return alpaka::onHost::getGlobalMemCapacityBytes();
#endif
    }

    /** Return the number of free bytes in a CPU domain.
     *
     * Linux-only implementation via /sys/devices/system/node/nodeX/meminfo. If the CPU domain has multiple attached
     * NUMA memory nodes, e.g. DDR and HBM, the returned value is the sum of all attached memory nodes.
     *
     * @param cpuDomainIdx Index of the cpu group.
     */
    inline size_t getFreeGlobalMemBytes(uint32_t cpuDomainIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(cpuDomainIdx == allDomains)
            return alpaka::onHost::getFreeGlobalMemBytes();

        size_t bytes = 0u;
        for(hwloc_obj_t const node : getMemoryNodes(getCpuDomainObj(cpuDomainIdx)))
        {
            auto const freeBytes = parseNodeMemInfoValueBytes(node->os_index, "MemFree:");
            if(!freeBytes.has_value())
            {
                throw std::runtime_error(
                    "Could not read per-node MemFree from /sys/devices/system/node/node"
                    + std::to_string(node->os_index) + "/meminfo");
            }
            bytes += *freeBytes;
        }

        return bytes;
#else
        alpaka::unused(cpuDomainIdx);
        return alpaka::onHost::getFreeGlobalMemBytes();
#endif
    }
} // namespace alpaka::onHost::internal::hwloc
