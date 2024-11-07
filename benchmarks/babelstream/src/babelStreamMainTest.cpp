
#include "babelStreamCommon.hpp"
#include "catch2/catch_session.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace alpaka;

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot.
 * Babelstream is a memory-bound benchmark since the main operation in the kernels has high Code Balance (bytes/FLOP)
 * value. For example c[i] = a[i] + b[i]; has 2 reads 1 writes and has one FLOP operation. For double precision each
 * read-write is 8 bytes. Hence Code Balance (3*8 / 1) = 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments:
 * ./babelstream --array-size=33554432 --number-runs=100
 * Runt with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and defaul arrary size and num runs:
 * ./babelstream --success
 * ./babelstream -r a.xml
 * Run with Custom and catch2 arguments together:
 * ./babelstream  --success --array-size=1280000 --number-runs=10
 * Help to list custom and catch2 arguments
 * ./babelstream -?
 * ./babelstream --help
 *  According to tests, 2^25 or larger data size values are needed for proper benchmarking:
 *  ./babelstream --array-size=33554432 --number-runs=100
 */

// Main function that integrates Catch2 and custom argument handling
int main(int argc, char* argv[])
{
    // Handle custom arguments
    handleCustomArguments(argc, argv);

    // Initialize Catch2 and pass the command-line arguments to it
    int result = Catch::Session().run(argc, argv);

    // Return the result of the tests
    return result;
}

//! Initialization kernel
struct InitKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param initA the value to set all items in the vector
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T* b, T* c, T initA) const
    {
#if 1
        auto const [i] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
#else
        for(auto [i] : IndependentDataIter{acc})
#endif
        {
            a[i] = initA;
            b[i] = static_cast<T>(0.0);
            c[i] = static_cast<T>(0.0);
        }
    }
};

//! Vector copying kernel
struct CopyKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T* b) const
    {
#if 0
        auto const [index] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        b[index] = a[index];
#elif 1
        for(auto [i] : IndependentDataIter{acc,Vec{1024u*1024u}})
        {
            b[i] = a[i];
        }
#else
        auto const [index] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        for(uint32_t i = index; i < (1024u * acc[layer::thread].count()).x();
            i += (256u * acc[layer::thread].count()).x())
            b[i] = a[i];
#endif
    }
};

//! Kernel multiplies the vector with a scalar, scaling or multiplication kernel
struct MultKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for result vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* const a, T* b) const
    {
        T const scalar = static_cast<T>(scalarVal);
#if 0
        auto const [i] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        b[i] = scalar * a[i];
#else
        for(auto [i] : IndependentDataIter{acc, Vec{1024u*1024u}})
        {
            b[i] = scalar * a[i];
        }
#endif
    }
};

//! Vector summation kernel
struct AddKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param c Pointer for result vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c) const
    {
#if 1
        auto const [i] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        c[i] = a[i] + b[i];
#else
        for(auto [i] : IndependentDataIter{acc})
        {
            c[i] = a[i] + b[i];
        }
#endif
    }
};

//! Kernel to find the linear combination of 2 vectors by initially scaling one of them
struct TriadKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param c Pointer for result vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c) const
    {
        T const scalar = static_cast<T>(scalarVal);
#if 1
        auto const [i] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        c[i] = a[i] + scalar * b[i];
#else
        for(auto [i] : IndependentDataIter{acc})
        {
            c[i] = a[i] + scalar * b[i];
        }
#endif
    }
};

//! Dot product of two vectors. The result is not a scalar but a vector of block-level dot products. For the
//! BabelStream implementation and documentation: https://github.com/UoB-HPC
struct DotKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param a Pointer for vector a
    //! \param b Pointer for vector b
    //! \param sum Pointer for result vector consisting sums for each block
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* sum, auto arraySize) const
    {
        auto& tbSum = alpaka::declareSharedVar<T[blockThreadExtentMain]>(acc);
#if 0
        auto [i] = acc[layer::block].idx() * acc[layer::thread].count() + acc[layer::thread].idx();
        auto const [local_i] = acc[layer::thread].idx();
        auto const [totalThreads] = acc[layer::thread].count() *  acc[layer::block].count();

        T threadSum = 0;
        for(; i < arraySize; i += totalThreads)
            threadSum += a[i] * b[i];
        tbSum[local_i] = threadSum;
#else

        T threadSum = 0;
        for(auto [i] : IndependentDataIter{acc, arraySize})
        {
            threadSum += a[i] * b[i];
        }
        for(auto [local_i] : IndependentBlockThreadIter{acc})
        {
            tbSum[local_i] = threadSum;
        }
#endif
#if 1
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            alpaka::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }
        if(local_i == 0)
            sum[acc[layer::block].idx().x()] = tbSum[local_i];
#else
        if(acc[layer::thread].idx().x() == 0)
        {
            auto registerSum = tbSum[0];
            for(uint32_t i = 1u; i < acc[layer::thread].count(); ++i)
                registerSum += tbSum[i];
            sum[acc[layer::block].idx().x()] = registerSum;
        }
#endif
    }
};

//! \brief The Function for testing babelstream kernels for given Acc type and data type.
//! \tparam TAcc the accelerator type
//! \tparam DataType The data type to differentiate single or double data type based tests.
template<typename DataType>
void testKernels(auto api)
{
    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device devAcc = platform.makeDevice(0);

    std::cout << getName(platform) << " " << devAcc.getName() << std::endl;

    // auto mapping = mapping::cpuBlockOmpThreadOne;
    auto possibleMappings = supportedMappings(devAcc);
    auto mapping = std::get<0>(possibleMappings);
    std::cout << "used mapping " << core::demangledName(mapping) << std::endl;


    // Meta data
    // A MetaData class instance to keep the problem and results to print later
    MetaData metaData;
    std::string dataTypeStr;
    if(std::is_same<DataType, float>::value)
    {
        dataTypeStr = "single";
    }
    else if(std::is_same<DataType, double>::value)
    {
        dataTypeStr = "double";
    }

    // Get the host device for allocating memory on the host
    Queue queue = devAcc.makeQueue();
    Platform platformHost = makePlatform(api::cpu);
    Device devHost = platformHost.makeDevice(0);

    using Idx = std::uint32_t;

    // Create vectors
    auto arraySize = Vec{static_cast<Idx>(arraySizeMain)};

    // Acc buffers
    auto bufAccInputA = alpaka::alloc<DataType>(devAcc, arraySize);
    auto bufAccInputB = alpaka::alloc<DataType>(devAcc, arraySize);
    auto bufAccOutputC = alpaka::alloc<DataType>(devAcc, arraySize);

    // Host buffer as the result
    auto bufHostOutputA = alpaka::alloc<DataType>(devHost, arraySize);
    auto bufHostOutputB = alpaka::alloc<DataType>(devHost, arraySize);
    auto bufHostOutputC = alpaka::alloc<DataType>(devHost, arraySize);

    // Create pointer variables for buffer access
    auto bufAccInputAPtr = std::data(bufAccInputA);
    auto bufAccInputBPtr = std::data(bufAccInputB);
    auto bufAccOutputCPtr = std::data(bufAccOutputC);

    auto numBlocks = arraySize / static_cast<Idx>(blockThreadExtentMain);
    auto dataBlocking = DataBlocking{numBlocks, Vec{static_cast<Idx>(blockThreadExtentMain)}};

    // Vector of average run-times of babelstream kernels
    std::vector<double> avgExecTimesOfKernels;
    std::vector<double> minExecTimesOfKernels;
    std::vector<double> maxExecTimesOfKernels;
    std::vector<std::string> kernelLabels;
    // Vector for collecting successive run-times of a single kernel in benchmark macro
    std::vector<double> times;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        for(auto i = 0; i < numberOfRuns; i++)
        {
            double runtime = 0.0;
            alpaka::wait(queue);
            auto start = std::chrono::high_resolution_clock::now();
            kernelFunc();
            alpaka::wait(queue);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            runtime = duration.count();
            times.push_back(runtime);
        }

        // find the minimum of the durations array.
        // In benchmarking the first item of the runtimes array is not included in calculations.
        const auto minmaxPair = findMinMax(times);
        minExecTimesOfKernels.push_back(minmaxPair.first);
        maxExecTimesOfKernels.push_back(minmaxPair.second);
        avgExecTimesOfKernels.push_back(findAverage(times));
        kernelLabels.push_back(kernelLabel);
        times.clear();
    };

    // Run kernels one by one
    // Test the init-kernel.
    measureKernelExec(
        [&]()
        {
            queue.enqueue(
                mapping,
                dataBlocking,
                InitKernel{},
                bufAccInputAPtr,
                bufAccInputBPtr,
                bufAccOutputCPtr,
                static_cast<DataType>(valA));
        },
        "InitKernel");

    auto dataBlockingCopy = DataBlocking{Vec{static_cast<Idx>(256)}, Vec{static_cast<Idx>(blockThreadExtentMain)}};

    // Test the copy-kernel. Copy A one by one to B.
    measureKernelExec(
        [&]() { queue.enqueue(mapping, dataBlockingCopy, CopyKernel(), bufAccInputAPtr, bufAccInputBPtr); },
        "CopyKernel");

    auto dataBlockingMult = DataBlocking{Vec{static_cast<Idx>(256)}, Vec{static_cast<Idx>(blockThreadExtentMain)}};
    // Test the scaling-kernel. Calculate B=scalar*A.
    measureKernelExec(
        [&]() { queue.enqueue(mapping, dataBlockingMult, MultKernel(), bufAccInputAPtr, bufAccInputBPtr); },
        "MultKernel");

    // Test the addition-kernel. Calculate C=A+B. Where B=scalar*A.
    measureKernelExec(
        [&]()
        { queue.enqueue(mapping, dataBlocking, AddKernel(), bufAccInputAPtr, bufAccInputBPtr, bufAccOutputCPtr); },
        "AddKernel");

    // Test the Triad-kernel. Calculate C=A+scalar*B where B=scalar*A.
    measureKernelExec(
        [&]()
        { queue.enqueue(mapping, dataBlocking, TriadKernel(), bufAccInputAPtr, bufAccInputBPtr, bufAccOutputCPtr); },
        "TriadKernel");


    // Copy arrays back to host
    alpaka::memcpy(queue, bufHostOutputC, bufAccOutputC);
    alpaka::memcpy(queue, bufHostOutputB, bufAccInputB);
    alpaka::memcpy(queue, bufHostOutputA, bufAccInputA);

    alpaka::wait(queue);

    // Verify the results
    //
    // Find sum of the errors as sum of the differences from expected values
    DataType initVal{static_cast<DataType>(0.0)};
    DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

    auto const expectedC = static_cast<DataType>(valA + scalarVal * scalarVal * valA);
    auto const expectedB = static_cast<DataType>(scalarVal * valA);
    auto const expectedA = static_cast<DataType>(valA);

    // sum of the errors for each array
    for(Idx i = 0; i < arraySize; ++i)
    {
        sumErrC += bufHostOutputC.data()[static_cast<Idx>(i)] - expectedC;
        sumErrB += bufHostOutputB.data()[static_cast<Idx>(i)] - expectedB;
        sumErrA += bufHostOutputA.data()[static_cast<Idx>(i)] - expectedA;
    }

    // Normalize and compare sum of the errors
    REQUIRE(FuzzyEqual(sumErrC / static_cast<DataType>(arraySize.x()) / expectedC, static_cast<DataType>(0.0)));
    REQUIRE(FuzzyEqual(sumErrB / static_cast<DataType>(arraySize.x()) / expectedB, static_cast<DataType>(0.0)));
    REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize.x()) / expectedA, static_cast<DataType>(0.0)));


    // Vector of sums of each block
    auto bufAccSumPerBlock = alpaka::alloc<DataType>(devAcc, dataBlocking.m_numBlocks);
    auto bufHostSumPerBlock = alpaka::alloc<DataType>(devHost, dataBlocking.m_numBlocks);

    auto dataBlockingDot = DataBlocking{Vec{static_cast<Idx>(256)}, Vec{static_cast<Idx>(blockThreadExtentMain)}};

    measureKernelExec(
        [&]()
        {
            queue.enqueue(
                mapping,
                dataBlockingDot,
                DotKernel(), // Dot kernel
                std::data(bufAccInputA),
                std::data(bufAccInputB),
                std::data(bufAccSumPerBlock),
                arraySize);
        },
        "DotKernel");

    alpaka::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock);
    alpaka::wait(queue);

    DataType const* sumPtr = std::data(bufHostSumPerBlock);
    auto const result = std::reduce(sumPtr, sumPtr + dataBlocking.m_numBlocks.x(), DataType{0});
    // Since vector values are 1, dot product should be identical to arraySize
    REQUIRE(FuzzyEqual(static_cast<DataType>(result), static_cast<DataType>(arraySize.x() * 2)));
    // Add workdiv to the list of workdivs to print later

    metaData.setItem(BMInfoDataType::WorkDivDot, dataBlocking);

    //
    // Calculate and Display Benchmark Results
    //
    std::vector<double> bytesReadWriteMB = {
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(3u, static_cast<unsigned>(arraySize)),
        getDataThroughput<DataType>(2u, static_cast<unsigned>(arraySize)),
    };

    // calculate the bandwidth as throughput per seconds
    std::vector<double> bandwidthsPerKernel;
    if(minExecTimesOfKernels.size() == kernelLabels.size())
    {
        for(size_t i = 0; i < minExecTimesOfKernels.size(); ++i)
        {
            bandwidthsPerKernel.push_back(calculateBandwidth(bytesReadWriteMB.at(i), minExecTimesOfKernels.at(i)));
        }
    }

    // Setting fields of Benchmark Info map. All information about benchmark and results are stored in a single
    // map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);


    metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);
    metaData.setItem(BMInfoDataType::WorkDivCopy, dataBlockingCopy);
    metaData.setItem(BMInfoDataType::WorkDivAdd, dataBlocking);
    metaData.setItem(BMInfoDataType::WorkDivMult, dataBlockingMult);
    metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);

    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, alpaka::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, core::demangledName(mapping));
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");
    // Join elements and create a comma separated string
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(bytesReadWriteMB, ", "));
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(bandwidthsPerKernel, ", "));
    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(minExecTimesOfKernels, ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(maxExecTimesOfKernels, ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(avgExecTimesOfKernels, ", "));

    // Print the summary as a table, if a standard serialization is needed other functions of the class can be
    // used
    std::cout << metaData.serializeAsTable() << std::endl;
}

using TestApis = std::decay_t<decltype(enabledApis)>;

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Five Kernels<Float>", "[benchmark-test]", TestApis)
{
    auto api = TestType{};
    // Run tests for the float data type
    testKernels<float>(api);
}

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Five Kernels<Double>", "[benchmark-test]", TestApis)
{
    auto api = TestType{};
    // Run tests for the double data type
    testKernels<double>(api);
}
