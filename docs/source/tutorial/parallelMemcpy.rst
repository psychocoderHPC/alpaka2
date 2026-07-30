.. _parallel-memcpy:

Parallel memcpy on CPU (host) backend
=====================================

Host-side optional parallel memcpy implementation which
enables kernel-backed, multi-threaded copies for large host-to-host / host-to-device /
device-to-host transfers on CPU backends.

The feature consists of:

- a build-time CMake option to set the default behaviour,
- runtime environment variables to override behavior per-process,
- a per-queue runtime configuration object (``alpaka::onHost::config::MemcpyConfig``) for per-queue control and a dedicated API,
- a small helper implementation that dispatches to either a sequential ``std::memcpy`` path or a kernel-based parallel copy depending on configuration and copy size.

Using this feature can improve performances of applications, especially if 
there is multiple large copy or a lot of memory exchange between host and device.

Build-time option
-----------------
A CMake option controls the compile-time default:

- ``alpaka_PARALLEL_MEMCPY_HOST_AS_DEFAULT`` (OFF by default)

When ``ON``, change default mode from "sequential" to "parallel". To enable
this at configure time:

.. code-block:: bash

  cmake -Dalpaka_PARALLEL_MEMCPY_HOST_AS_DEFAULT=ON <other-cmake-args> ..

This set the compile time macro:
- ``ALPAKA_PARALLEL_MEMCPY_HOST_AS_DEFAULT``, resolves to ``true`` or ``false`` depending on the CMake option. If undefined, the implementation uses ``false`` as default.

Note: this only affects the compile-time default used when each Queue is
constructed. It can still be overridden at process start via environment
variables (see below) or changed at runtime using the queue API.


Runtime configuration (environment variables)
---------------------------------------------
At runtime a process-wide environment variable may configure how newly-created
queues initialize their memcpy configuration. This can be used to set/override 
default behavior set in CMake Configuration. 
The following environment variables are read at Queue creation time:

- ``ALPAKA_MEMCPY_MODE`` :

  - values: ``sequential`` (or ``serial``) | ``parallel``
  - If not set, the queue uses the compile-time default given by the macro.
  - Invalid values cause a std::invalid_argument to be thrown during parsing.

- ``ALPAKA_MEMCPY_NUM_CORES`` :

  - integer >= 0
  - Sets the number of cores used for parallel memcpy. ``0`` means all available cores (default).
  - Values above the number of cores of the numa domain of the queue are silently capped to it. "Cores" are
    logical CPUs, SMT siblings are counted too.
  - The value sets the number of chunks the copy is split into, the number of cores used is an indirect
    consequence of that: the chunks are enqueued as thread blocks and distributed over the thread pool of the
    backend, which this value does not size.

- ``ALPAKA_MEMCPY_MIN_SIZE`` :

  - integer >= 0 (bytes)
  - Minimum total copy size (in bytes) before parallel memcpy is chosen.
  - Default: ``1048576`` (1 MiB). Copies smaller than this threshold use the sequential pathway even if the mode is ``parallel``.

Examples:

.. code-block:: bash

  # Force parallel mode for newly created queues in this process
  export ALPAKA_MEMCPY_MODE=parallel

  # Limit parallel memcpy to at most 4 cores
  export ALPAKA_MEMCPY_NUM_CORES=4

  # Only parallelize copies of 2 MiB or more
  export ALPAKA_MEMCPY_MIN_SIZE=$((2*1024*1024))

Per-queue API
-------------
Each CPU queue owns a ``MemcpyConfig`` instance that is created when the queue
is constructed. This per-queue object is thread-safe and can be used to inspect
and change the configuration for that queue without affecting other queues.

Relevant functionality :

- ``queue.get()->getMemcpyConfig()``:

  - Returns a (mutable) reference to that queue's ``MemcpyConfig``. ``onHost::Queue`` is a handle, the method is
    reached through ``get()``.
  - The ``hasMemcpyConfig`` trait in ``parallelMemcpy.hpp`` guards use in templates: ``alpaka::onHost::config::hasMemcpyConfig<MyQueueType>`` is true for CPU queues.

- ``onHost::config::MemcpyMode```

  - sequential / parallel

- ``alpaka::onHost::config::MemcpyConfig``:

  - getMode() / setMode(MemcpyMode)
  - getNumCores() / setNumCores(uint32_t)
  - getMinSizeForParallel() / setMinSizeForParallel(size_t)
  - reset() - reset to compile-time defaults or re-parse environment variable.
  - getConfigString() - human-readable description.


API usage examples
------------------
Change mode/configuration at runtime (affects subsequent memcpys on that queue) 
and query and print the config for a queue:

.. literalinclude:: ../../snippets/example/240_parallelMemcpy.cpp
    :language: cpp
    :start-after: BEGIN-TUTORIAL-parallelMemcpy
    :end-before: END-TUTORIAL-parallelMemcpy
    :dedent:


Notes & troubleshooting
-----------------------
- If an invalid environment variable is set (non-numeric, wrong string, negative value),
  the config parsing will throw ``std::invalid_argument`` during queue initialization.
  Make sure environment variable values are valid.
- Setting the compile-time default (CMake option) only changes the initial state of
  newly-created queues; processes may still override via environment variables.
- The work is distributed over the bytes of the copy and not over its elements or rows, therefore a copy with few
  rows, e.g. ``[4][8Mi]``, still uses all chunks. A chunk can begin and end in the middle of a row.

Complete Source File
--------------------

.. raw:: html

   <details class="full-source">
   <summary>240_parallelMemcpy.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/240_parallelMemcpy.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
