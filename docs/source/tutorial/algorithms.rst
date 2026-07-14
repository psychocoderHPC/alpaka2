Host Algorithms
===============

*alpaka* provides host-side algorithms that execute on the selected :ref:`backend <backend>` through an *alpaka* queue and :ref:`executor <executor>`.
They operate on *alpaka* `buffers, views, MdSpan <../doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IMdSpan.html>`_ and `generators <../doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IDataSource.html>`_.
You can always skip the :ref:`executor`, in this case *alpaka* is using a fitting executor depending on the API of the backend.
All algorithms except ``scan`` can operate on n-dimensional data.
They are implemented to use `SIMD data packs <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`_ to improve the memory bandwidth and compute utilization.
SIMD is often explicitly exposed to provide you the full control and optimization potential.

iota
----

``onHost::iota`` fills one or more output buffers with a linear sequence of scalar indies.
For multidimensional buffers, the linear value increases fastest in the last dimension.
In this example we start the enumeration with 10.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-iota
    :end-before: END-TUTORIAL-iota
    :dedent:

Reduction
---------

Reduction writes its result into the first element of the output buffer.
The neutral element of the operation binary reduce functor must be provided explicitly.
This allows keeping the data in device memory without going at any time to host memory.
That is different compared to ``std::reduce``.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-reduce
    :end-before: END-TUTORIAL-reduce
    :dedent:

Transform
---------

``onHost::transform`` is the host-side algorithm equivalent of an element-wise kernel.
It applies a functor to one or more inputs and writes the result into an output buffer.
The data type of the input and output buffers can differ.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformFunctor
    :end-before: END-TUTORIAL-transformFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformCall
    :end-before: END-TUTORIAL-transformCall
    :dedent:

ScalarFunc and StencilFunc
~~~~~~~~~~~~~~~~~~~~~~~~~~

The functor can be wrapped to control how it is executed:

- ``ScalarFunc`` forces scalar (element-wise) execution even when the algorithm could use SIMD packs internally.
  Use this when your functor uses operations that do not work on SIMD types (e.g., branching, ``math::min``, ``math::max``).

- ``StencilFunc`` signals that the functor accepts ``SimdPtr`` arguments.
  A ``SimdPtr`` supports ``operator[]`` for relative indexing, enabling stencil operations.
  When you use ``StencilFunc``, all arguments are passed as ``SimdPtr``.
  You must ensure memory accesses stay in bounds — typically by operating on a sub-view that excludes the halo.
  See the unit tests for a complete stencil example.

- For all transform algorithms, not wrapping the transform functor as ``ScalarFunc`` or ``StencilFunc`` requires the functor to accept ``alpaka::Simd`` as its input arguments. This will be shown in the next example.


Transform-Reduce
----------------

``transformReduce`` combines a data transformation step with a reduction step.
It is used for dot products, weighted sums, norms, and many "compute a value per element and then accumulate it" patterns.
The first functor is the binary reduction operator. The second one is the element-wise transform and must accept as many arguments as there are input views/MdSpans passed to ``onHost::transformReduce``.
Compared to ``std::transform_reduce`` the function support any amount of input views/MdSpans.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceFunctor
    :end-before: END-TUTORIAL-transformReduceFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-transformReduceCall
    :end-before: END-TUTORIAL-transformReduceCall
    :dedent:

Writing to input arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both ``transform`` and ``transformReduce`` can write to their input arguments when the functor is wrapped in ``StencilFunc``.
The ``SimdPtr`` passed to a stencil functor supports assignment (``operator=``), allowing in-place modification of the input data.
This is useful for algorithms that need to update their inputs as they process them.
However, the user is responsible for ensuring no data races occur — the algorithm does not enforce ordering between reads and writes to the same element.

Concurrent
----------

``onHost::concurrent`` executes an n-ary functor on each element of all input/output buffers.
This lets you implement a transform with a free number of outputs, or fuse multiple reads and writes into a single kernel launch.
The functor receives ``SimdPtr`` arguments, so stencil-style indexing works without wrapping in ``StencilFunc``.
You can also pass generators as inputs (see unit tests for examples).

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-concurrentFunctor
    :end-before: END-TUTORIAL-concurrentFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-concurrentCall
    :end-before: END-TUTORIAL-concurrentCall
    :dedent:

Scan
----

Unlike the other algorithms in this chapter, the current scan implementation is restricted to one-dimensional data.
That fits common `prefix-sum <https://en.wikipedia.org/wiki/Prefix_sum>`_ use cases such as offsets, compaction maps, and cumulative counters, where the logical input is already a linear sequence.
This examples uses an explicit temporary storage where the size is provided via ``getScanBufferSize``, this can be used to optimize the performance in cases where scan is called multiple times.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-scan
    :end-before: END-TUTORIAL-scan
    :dedent:

Generators Instead of Input Buffers
-----------------------------------

Several alpaka algorithms also accept generators as inputs.
That is useful when one input is synthetic, such as a linear index, and you do not want to materialize another buffer just to hold it.
``LinearizedIdxGenerator`` generates scalar indexes from n-dimensional indexes and behaves like a `IDataSource <../doxygen/conceptalpaka_1_1concepts_1_1impl_1_1IDataSource.html>`_.
It behaves like a virtual buffer whose value at each position is the corresponding linear index.
In this example, we add a value from one input buffer to the generated index.

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorFunctor
    :end-before: END-TUTORIAL-generatorFunctor
    :dedent:

  .. literalinclude:: ../../snippets/example/130_algorithms.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-generatorCall
    :end-before: END-TUTORIAL-generatorCall
    :dedent:

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>130_algorithms.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/130_algorithms.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
