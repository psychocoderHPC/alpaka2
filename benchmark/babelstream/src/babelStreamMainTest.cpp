
#include "babelStreamCommon.hpp"
#include "catch2/catch_session.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <string>

using namespace alpaka;

/**
 * Babelstream benchmarking example. Babelstream has 5 kernels. Add, Multiply, Copy, Triad and Dot. NStream is
 * optional. Init kernel is run before 5 standard kernel sequence. Babelstream is a memory-bound benchmark since the
 * main operation in the kernels has high Code Balance (bytes/FLOP) value. For example c[i] = a[i] + b[i]; has 2 reads
 * 1 writes and has one FLOP operation. For double precision each read-write is 8 bytes. Hence Code Balance (3*8 / 1) =
 * 24 bytes/FLOP.
 *
 * Some implementations and the documents are accessible through https://github.com/UoB-HPC
 *
 * Can be run with custom arguments as well as catch2 arguments
 * Run with Custom arguments and for kernels: init, copy, mul, add, triad (and dot kernel if a multi-thread acc
 * available):
 * ./babelstream --array-size=33554432 --number-runs=100
 * Run with Custom arguments and select from 3 kernel groups: all, triad, nstream
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=triad (only triad kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=nstream (only nstream kernel)
 * ./babelstream --array-size=33554432 --number-runs=100 --run-kernels=all (default case. Add, Multiply, Copy, Triad
 * and Dot) Run with default array size and num runs:
 * ./babelstream
 * Run with Catch2 arguments and default array size and num runs:
 * ./babelstream --success
 * ./babelstream -r xml
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
    //! \param initialA the value to set all items in the vector a
    //! \param initialB the value to set all items in the vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T* a,
        T* b,
        T* c,
        T initialA,
        T initialB,
        T initialC,
        auto arraySize) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
        {
            a[i] = initialA;
            b[i] = initialB;
            c[i] = initialC;
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
    //! \param c Pointer for vector c
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T* c, auto arraySize) const
    {
        for(auto [index] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
            c[index] = a[index];
    }
};

//! Kernel multiplies the vector with a scalar, scaling or multiplication kernel
struct MultKernel
{
    //! The kernel entry point
    //! \tparam TAcc The accelerator environment to be executed on.
    //! \tparam T The data type
    //! \param acc The accelerator to be executed on.
    //! \param c Pointer for vector c
    //! \param b Pointer for result vector b
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* b, T* const c, auto arraySize) const
    {
        T const scalar = static_cast<T>(scalarVal);
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
            b[i] = scalar * c[i];
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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* c, auto arraySize) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
            c[i] = a[i] + b[i];
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
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c, auto arraySize) const
    {
        T const scalar = static_cast<T>(scalarVal);
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
            a[i] = b[i] + scalar * c[i];
    }
};

//! Optional kernel, not one of the 5 standard Babelstream kernels
struct NstreamKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T* a, T const* b, T const* c, auto arraySize) const
    {
        T const scalar = static_cast<T>(scalarVal);
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
            a[i] += b[i] + scalar * c[i];
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
    //! \param sum Pointer for result vector consisting sums of blocks
    //! \param arraySize the size of the array
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* a, T const* b, T* sum, auto arraySize) const
    {
        auto tbSum = onAcc::declareSharedMdArray<T>(acc, CVec<uint32_t, blockThreadExtentMain>{});
#if 1
        auto numFrames = Vec{acc[frame::count]};
        auto frameExtent = Vec{acc[frame::extent]};

        auto traverseInFrame = onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{frameExtent});
        /* init shared memory
         * no need to synchronize because after the loop because the same index map will be used to access the shared
         * memory in the tribble loop.
         */
        for(auto [elemIdxInFrame] : traverseInFrame)
        {
            tbSum[elemIdxInFrame] = T{0};
        }

        auto traverseOverFrames = onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numFrames});

        for(auto frameIdx : traverseOverFrames)
        {
            for(auto elemIdxInFrame : traverseInFrame)
            {
                for(auto [i] : onAcc::makeIdxMap(
                        acc,
                        onAcc::iter::WorkerGroup{frameIdx * frameExtent + elemIdxInFrame, numFrames * frameExtent},
                        IdxRange{arraySize}))
                {
                    tbSum[elemIdxInFrame] += a[i] * b[i];
                }
            }
        }

        // aggregate for each thread but skip the first shared memory slot
        for(auto [elemIdxInFrame] :
            onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{acc[layer::thread].count(), frameExtent}))
        {
            tbSum[acc[layer::thread].idx()] += tbSum[elemIdxInFrame];
        }

#else
        // this version is shorter in code but runs into floating point stability issues due to to long aggregation of
        // value by a single thread
        auto threadSum = T{0};
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{arraySize}))
        {
            threadSum += a[i] * b[i];
        }
        for(auto [local_i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::threadsInBlock))
        {
            tbSum[local_i] = threadSum;
        }
#endif
        auto const [local_i] = acc[layer::thread].idx();
        auto const [blockSize] = acc[layer::thread].count();
        for(auto offset = blockSize / 2; offset > 0; offset /= 2)
        {
            onAcc::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }
        if(local_i == 0)
            sum[acc[layer::block].idx().x()] = tbSum[local_i];
    }
};

//! \brief The Function for testing babelstream kernels for given Acc type and data type.
//! \tparam TAcc the accelerator type
//! \tparam DataType The data type to differentiate single or double data type based tests.
template<typename DataType>
void testKernels(auto cfg)
{
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        std::cout << "Kernels: Init, Copy, Mul, Add, Triad, Dot Kernels" << std::endl;
    }

    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    onHost::Platform platform = onHost::makePlatform(api);
    onHost::Device devAcc = platform.makeDevice(0);

    std::cout << getName(platform) << "\n" << getDeviceProperties(devAcc) << std::endl;

    std::cout << "used exec " << core::demangledName(exec) << std::endl;

    // A MetaData class instance to keep the benchmark info and results to print later. Does not include intermediate
    // runtime data.
    BenchmarkMetaData metaData;

    // Convert data-type to string to display
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
    onHost::Queue queue = devAcc.makeQueue();
    onHost::Platform platformHost = onHost::makePlatform(api::cpu);
    onHost::Device devHost = platformHost.makeDevice(0);

    // Create vectors
    using Idx = std::uint32_t;
    auto arraySize = Vec{static_cast<Idx>(arraySizeMain)};

    // Acc buffers
    auto bufAccInputA = onHost::alloc<DataType>(devAcc, arraySize);
    auto bufAccInputB = onHost::allocMirror(devAcc, bufAccInputA);
    auto bufAccOutputC = onHost::allocMirror(devAcc, bufAccInputA);

    // Host buffer as the result
    auto bufHostOutputA = onHost::allocMirror(devHost, bufAccInputA);
    auto bufHostOutputB = onHost::allocMirror(devHost, bufAccInputB);
    auto bufHostOutputC = onHost::allocMirror(devHost, bufAccOutputC);

    auto numBlocks = arraySize / static_cast<Idx>(blockThreadExtentMain);
    auto dataBlocking = alpaka::DataBlocking{numBlocks, Vec{static_cast<Idx>(blockThreadExtentMain)}};

    // To record runtime data generated while running the kernels
    RuntimeResults runtimeResults;

    // Lambda for measuring run-time
    auto measureKernelExec = [&](auto&& kernelFunc, [[maybe_unused]] auto&& kernelLabel)
    {
        double runtime = 0.0;
        onHost::wait(queue);
        auto start = std::chrono::high_resolution_clock::now();
        kernelFunc();
        onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // get duration in seconds
        std::chrono::duration<double> duration = end - start;
        runtime = duration.count();
        runtimeResults.kernelToRundataMap[kernelLabel]->timingsSuccessiveRuns.push_back(runtime);
    };


    // Initialize logger before running kernels
    // Runtime result initialisation to be filled by each kernel
    runtimeResults.addKernelTimingsVec("InitKernel");
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        runtimeResults.addKernelTimingsVec("CopyKernel");
        runtimeResults.addKernelTimingsVec("AddKernel");
        runtimeResults.addKernelTimingsVec("TriadKernel");
        runtimeResults.addKernelTimingsVec("MultKernel");
        runtimeResults.addKernelTimingsVec("DotKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        runtimeResults.addKernelTimingsVec("NStreamKernel");
    }
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        runtimeResults.addKernelTimingsVec("TriadKernel");
    }


    // Init kernel
    measureKernelExec(
        [&]()
        {
            queue.enqueue(
                exec,
                dataBlocking,
                KernelBundle{
                    InitKernel{},
                    std::data(bufAccInputA),
                    std::data(bufAccInputB),
                    std::data(bufAccOutputC),
                    static_cast<DataType>(initA),
                    static_cast<DataType>(initB),
                    static_cast<DataType>(initC),
                    arraySize});
        },
        "InitKernel");

    // Init kernel will be run for all cases therefore add it to metadata unconditionally
    metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);

    // Dot kernel result
    DataType resultDot = static_cast<DataType>(0.0);

    // Main for loop to run the kernel-sequence
    for(auto i = 0; i < numberOfRuns; i++)
    {
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            // Test the copy-kernel. Copy A one by one to C.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        exec,
                        dataBlocking,
                        KernelBundle{CopyKernel(), std::data(bufAccInputA), std::data(bufAccOutputC), arraySize});
                },
                "CopyKernel");

            // Test the scaling-kernel. Calculate B=scalar*C. Where C = A.
            measureKernelExec(
                [&]() {
                    queue.enqueue(
                        exec,
                        dataBlocking,
                        MultKernel(),
                        std::data(bufAccInputB),
                        std::data(bufAccOutputC),
                        arraySize);
                },
                "MultKernel");

            // Test the addition-kernel. Calculate C=A+B. Where B=scalar*C or B=scalar*A.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        exec,
                        dataBlocking,
                        KernelBundle{
                            AddKernel(),
                            std::data(bufAccInputA),
                            std::data(bufAccInputB),
                            std::data(bufAccOutputC),
                            arraySize});
                },
                "AddKernel");
        }
        // Triad kernel is run for 2 command line arguments
        if(kernelsToBeExecuted == KernelsToRun::All || kernelsToBeExecuted == KernelsToRun::Triad)
        {
            // Test the Triad-kernel. Calculate A=B+scalar*C. Where C is A+scalar*A.
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        exec,
                        dataBlocking,
                        KernelBundle{
                            TriadKernel(),
                            std::data(bufAccInputA),
                            std::data(bufAccInputB),
                            std::data(bufAccOutputC),
                            arraySize});
                },
                "TriadKernel");
        }
        if(kernelsToBeExecuted == KernelsToRun::All)
        {
            auto dataBlockingDot = DataBlocking{
                Vec{static_cast<Idx>(dotGridBlockExtent)},
                Vec{static_cast<Idx>(blockThreadExtentMain)}};

            // Vector of sums of each block
            auto bufAccSumPerBlock = onHost::alloc<DataType>(devAcc, dataBlockingDot.m_numBlocks);
            auto bufHostSumPerBlock = onHost::allocMirror(devHost, bufAccSumPerBlock);


            // Test Dot kernel with specific blocksize which is larger than one
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        exec,
                        dataBlockingDot,
                        KernelBundle{
                            DotKernel(), // Dot kernel
                            std::data(bufAccInputA),
                            std::data(bufAccInputB),
                            std::data(bufAccSumPerBlock),
                            arraySize});
                    onHost::memcpy(queue, bufHostSumPerBlock, bufAccSumPerBlock);
                    onHost::wait(queue);
                    DataType const* sumPtr = std::data(bufHostSumPerBlock);
                    resultDot = std::reduce(sumPtr, sumPtr + dataBlockingDot.m_numBlocks.x(), DataType{0});
                },
                "DotKernel");

            // Add workdiv to the list of workdivs to print later
            metaData.setItem(BMInfoDataType::WorkDivDot, dataBlockingDot);
        }
        // NStream kernel is run only for one command line argument
        if(kernelsToBeExecuted == KernelsToRun::NStream)
        {
            // Test the NStream-kernel. Calculate A += B + scalar * C;
            measureKernelExec(
                [&]()
                {
                    queue.enqueue(
                        exec,
                        dataBlocking,
                        KernelBundle{
                            NstreamKernel(),
                            std::data(bufAccInputA),
                            std::data(bufAccInputB),
                            std::data(bufAccOutputC),
                            arraySize});
                },
                "NStreamKernel");
        }
        onHost::wait(queue);
    } // End of MAIN LOOP which runs the kernels many times


    // Copy results back to the host, measure copy time
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Copy arrays back to host since the execution of kernels except dot kernel finished
        onHost::memcpy(queue, bufHostOutputC, bufAccOutputC);
        onHost::memcpy(queue, bufHostOutputB, bufAccInputB);
        onHost::memcpy(queue, bufHostOutputA, bufAccInputA);
        onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        // Get duration in seconds
        std::chrono::duration<double> duration = end - start;
        double copyRuntime = duration.count();
        metaData.setItem(BMInfoDataType::CopyTimeFromAccToHost, copyRuntime);
    }

    //
    // Result Verification and BW Calculation for 3 cases
    //

    // Generated expected values by doing the same chain of operations due to floating point error
    DataType expectedA = static_cast<DataType>(initA);
    DataType expectedB = static_cast<DataType>(initB);
    DataType expectedC = static_cast<DataType>(initC);

    // To calculate expected results by applying at host the same operation sequence
    calculateBabelstreamExpectedResults(expectedA, expectedB, expectedC);

    // Verify the resulting data, if kernels are init, copy, mul, add, triad and dot kernel
    if(kernelsToBeExecuted == KernelsToRun::All)
    {
        // Find sum of the errors as sum of the differences from expected values
        constexpr DataType initVal{static_cast<DataType>(0.0)};
        DataType sumErrC{initVal}, sumErrB{initVal}, sumErrA{initVal};

        // sum of the errors for each array
        for(Idx i = 0; i < arraySize; ++i)
        {
            sumErrC += std::fabs(std::data(bufHostOutputC)[static_cast<Idx>(i)] - expectedC);
            sumErrB += std::fabs(std::data(bufHostOutputB)[static_cast<Idx>(i)] - expectedB);
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        // Normalize and compare sum of the errors
        // Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(sumErrC / static_cast<DataType>(arraySize.x()), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrB / static_cast<DataType>(arraySize.x()), static_cast<DataType>(0.0)));
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize.x()), static_cast<DataType>(0.0)));
        onHost::wait(queue);

        // Verify Dot kernel
        DataType const expectedSum = static_cast<DataType>(arraySize.x()) * expectedA * expectedB;
        //  Dot product should be identical to arraySize*valA*valB
        //  Use a different equality check if floating point errors exceed precision of FuzzyEqual function
        REQUIRE(FuzzyEqual(static_cast<float>(std::fabs(resultDot - expectedSum) / expectedSum), 0.0f));

        // Set workdivs of benchmark metadata to be displayed at the end
        metaData.setItem(BMInfoDataType::WorkDivInit, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivCopy, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivAdd, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivMult, dataBlocking);
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the Triad Kernel result if "--run-kernels=triad".
    else if(kernelsToBeExecuted == KernelsToRun::Triad)
    {
        // Verify triad by summing the error
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize.x(); ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }

        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize.x()) / expectedA, static_cast<DataType>(0.0)));
        metaData.setItem(BMInfoDataType::WorkDivTriad, dataBlocking);
    }
    // Verify the NStream Kernel result if "--run-kernels=nstream".
    else if(kernelsToBeExecuted == KernelsToRun::NStream)
    {
        auto sumErrA = static_cast<DataType>(0.0);
        // sum of the errors for each array
        for(Idx i = 0; i < arraySize.x(); ++i)
        {
            sumErrA += std::fabs(std::data(bufHostOutputA)[static_cast<Idx>(i)] - expectedA);
        }
        REQUIRE(FuzzyEqual(sumErrA / static_cast<DataType>(arraySize.x()) / expectedA, static_cast<DataType>(0.0)));

        metaData.setItem(BMInfoDataType::WorkDivNStream, dataBlocking);
    }

    // Runtime results of the benchmark: Calculate throughput and bandwidth
    // Set throuput values depending on the kernels
    runtimeResults.initializeByteReadWrite<DataType>(arraySize.x());
    runtimeResults.calculateBandwidthsForKernels<DataType>();

    // Set metadata to display all benchmark related information.
    //
    // All information about benchmark and results are stored in a single map
    metaData.setItem(BMInfoDataType::TimeStamp, getCurrentTimestamp());
    metaData.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    metaData.setItem(BMInfoDataType::DataSize, std::to_string(arraySizeMain));
    metaData.setItem(BMInfoDataType::DataType, dataTypeStr);
    // Device and accelerator
    metaData.setItem(BMInfoDataType::DeviceName, onHost::getName(devAcc));
    metaData.setItem(BMInfoDataType::AcceleratorType, core::demangledName(exec));
    // XML reporter of catch2 always converts to Nano Seconds
    metaData.setItem(BMInfoDataType::TimeUnit, "Nano Seconds");

    // get labels from the map
    std::vector<std::string> kernelLabels;
    std::transform(
        runtimeResults.kernelToRundataMap.begin(),
        runtimeResults.kernelToRundataMap.end(),
        std::back_inserter(kernelLabels),
        [](auto const& pair) { return pair.first; });
    // Join elements and create a comma separated string and set item
    metaData.setItem(BMInfoDataType::KernelNames, joinElements(kernelLabels, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> values(runtimeResults.getThroughputKernelArray());
    metaData.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(values, ", "));
    // Join elements and create a comma separated string and set item
    std::vector<double> valuesBW(runtimeResults.getBandwidthKernelVec());
    metaData.setItem(BMInfoDataType::KernelBandwidths, joinElements(valuesBW, ", "));

    metaData.setItem(BMInfoDataType::KernelMinTimes, joinElements(runtimeResults.getMinExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelMaxTimes, joinElements(runtimeResults.getMaxExecTimeKernelArray(), ", "));
    metaData.setItem(BMInfoDataType::KernelAvgTimes, joinElements(runtimeResults.getAvgExecTimeKernelArray(), ", "));
    // Print the summary as a table, if a standard serialization is needed other functions of the class can be used
    std::cout << metaData.serializeAsTable() << std::endl;
}

using TestApis = std::decay_t<decltype(onHost::allExecutorsAndApis(onHost::enabledApis))>;

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Float>", "[benchmark-test]", TestApis)
{
    auto apiAndExecutors = TestType::makeDict();
    // Run tests for the float data type
    testKernels<float>(apiAndExecutors);
}

// Run for all Accs given by the argument
TEMPLATE_LIST_TEST_CASE("TEST: Babelstream Kernels<Double>", "[benchmark-test]", TestApis)
{
    auto apiAndExecutors = TestType::makeDict();
    // Run tests for the double data type
    testKernels<double>(apiAndExecutors);
}
