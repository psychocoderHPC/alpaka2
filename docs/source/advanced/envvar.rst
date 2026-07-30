Environment Variables
=====================

LOG
---

``ALPAKA_LOG_DYNAMIC_LVL``
  .. code-block:: markdown

    Sets the runtime log level. Only works if ``alpaka_LOG=dynamic`` has been set in CMake.

  See :ref:`dev-logging`

Memcpy
------

``ALPAKA_MEMCPY_MODE``
  .. code-block:: markdown

    Sets the memcpy mode. Can be ``parallel`` or ``sequential``. Raises ``std::invalid_argument`` if the value is invalid.

``ALPAKA_MEMCPY_NUM_CORES``
  .. code-block:: markdown

    Sets the number of cores used for parallel memcpy. ``0`` means all available cores. Raises ``std::invalid_argument`` if the value is not an integer greater than or equal to ``0``. Values above the number of cores of the numa domain of the queue are silently capped to it. The value sets the number of chunks the copy is split into, the number of cores used is an indirect consequence of that: the chunks are distributed over the thread pool of the backend, which this value does not size.

``ALPAKA_MEMCPY_MIN_SIZE``
  .. code-block:: markdown

    Sets the minimum size (in bytes) required to use parallel memcpy. Must be an integer greater than or equal to ``0``, else raises ``std::invalid_argument``.

  See :ref:`parallel-memcpy`
