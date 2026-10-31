# Protocol

`protocol` defines macros and constants used in NVXSched.
It also provides helper functions to read these values.

The following configurations can be changed with environment variables.
Set them before starting the application; the variables retain the `XSCHED_` prefix from XSched.

## XQueue configuration

| Name                              | Type   | Range           | Default Value | Description                                                                         |
| --------------------------------- | ------ | --------------- | ------------- | ----------------------------------------------------------------------------------- |
| XSCHED_AUTO_XQUEUE                | string | ON/OFF          | OFF           | If it automatically creates XQueue when creating a hardware queue.                  |
| XSCHED_AUTO_XQUEUE_LEVEL          | int    | [1, 3]          | 1             | XQueue preemption level.                                                            |
| XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT | string | See below       | ALLOCATION    | Method used to construct memory contexts.                                           |
| XSCHED_AUTO_XQUEUE_THRESHOLD      | int    | [1, MAX_INT64]  | 16            | Maximum number of outstanding launched commands tracked by XQueue.                  |
| XSCHED_AUTO_XQUEUE_BATCH_SIZE     | int    | [1, threshold]  | 8             | Target spacing, in commands, between hardware synchronization points.               |
| XSCHED_AUTO_XQUEUE_PRIORITY       | int    | [-255, 255]     | 0             | Queue priority. Bigger values mean higher priority.                                 |
| XSCHED_AUTO_XQUEUE_UTILIZATION    | int    | [0, 100]        | unset         | Initial queue utilization weight for UP.                                            |
| XSCHED_AUTO_XQUEUE_TIMESLICE      | int    | [100, 10000000] | 5000          | Round budget for UP or average process timeslice for PUP/SPUP, in microseconds.     |
| XSCHED_AUTO_XQUEUE_DEADLINE       | int    | [1, MAX_INT64]  | unset         | KEDF deadline relative to the queue's ready time, in microseconds.                  |
| XSCHED_AUTO_XQUEUE_KDEADLINE      | int    | [1, MAX_INT64]  | unset         | Number of queues KEDF may run concurrently.                                         |
| XSCHED_AUTO_XQUEUE_LAXITY         | int    | [1, MAX_INT64]  | unset         | Delay from the queue's ready time before LAX changes its priority, in microseconds. |

Scheduling hints take effect according to the selected policy.
With `UP`, set `XSCHED_AUTO_XQUEUE_UTILIZATION` to give the queue a weight and include it in the policy's rotation.
`PUP` and `SPUP` assign weights to processes instead, using a default weight of 100; change these weights with process hints such as `xcli hint --pid PID -u 50`.
For `LAX`, `XSCHED_AUTO_XQUEUE_LAXITY` specifies how long the queue may wait after becoming ready: its priority is initially 0 and changes to `XSCHED_AUTO_XQUEUE_PRIORITY` when that delay expires.

## Platform Library Overrides

Platform adapters normally find their native runtime libraries from the system
search path and common installation paths. These variables can override the
library path when needed:

| Name                 |
| -------------------- |
| XSCHED_ASCEND_LIB    |
| XSCHED_COREX_LIB     |
| XSCHED_COREX_RT_LIB  |
| XSCHED_CUDA_LIB      |
| XSCHED_CUPTI_LIB     |
| XSCHED_CUDART_LIB    |
| XSCHED_CUDLA_LIB     |
| XSCHED_HIP_LIB       |
| XSCHED_LEVELZERO_LIB |
| XSCHED_OPENCL_LIB    |
| XSCHED_VPI_LIB       |

## CUDA XQueue-Specific Configuration

| Name                                  | Type   | Range    | Default Value | Description                                                                                                            |
| ------------------------------------- | ------ | -------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS | string | ON/OFF   | OFF           | Use one internal CUDA stream per process.                                                                              |
| XSCHED_CUDA_LV3_IMPL                  | string | TSG/TRAP | TRAP          | Preemption method of Level-3 XQueue for CUDA.                                                                          |
| XSCHED_CUDA_UVM                       | string | ON/OFF   | OFF           | Select the Level-1 backend unless TSG is selected explicitly.                                                          |
| XSCHED_CUDA_PTDS_AUTO_XQUEUE          | string | ON/OFF   | ON            | Create XQueues for per-thread default streams when automatic queues are enabled; does not apply in single-stream mode. |
| XSCHED_CUDA_SINGLE_PTDS_PER_PROCESS   | string | ON/OFF   | OFF           | Route per-thread default streams to the shared stream in single-stream mode.                                           |

For the Level-1 configuration used in the CUDA examples, set `XSCHED_CUDA_UVM=ON`.
To use TSG preemption, also select Level-3 and set `XSCHED_CUDA_LV3_IMPL=TSG`; the policy must then suspend and resume all queues sharing a CUDA context together.
These options control queue execution, while the shim redirects supported device allocations to managed memory independently.
See the [CUDA guide](../platforms/cuda/README.md#supported-preemption-level) for supported preemption levels.

## Level Zero XQueue-Specific Configuration

| Name                       | Type | Default Value    | Description                                           |
| -------------------------- | ---- | ---------------- | ----------------------------------------------------- |
| XSCHED_LEVELZERO_SLICE_CNT | int  | platform default | Command slicing count used by the Level Zero adapter. |

## Scheduler configuration

Change the scheduler type by setting `XSCHED_SCHEDULER`.
The default value is `APP`.

| Value | Full Name           | Transparency | Description                                                          |
| ----- | ------------------- | ------------ | -------------------------------------------------------------------- |
| LCL   | Local Scheduler     | ✅           | Run a local policy for this process; also set a valid XSCHED_POLICY. |
| GLB   | Global Scheduler    | ✅           | The process uses the global scheduler in XServer.                    |
| APP   | Application Managed | ❌           | The application uses XQueue APIs to manage XQueues directly.         |

By default, a scheduling operation suspends a queue without waiting for hardware synchronization or for all submitted work to complete.
The following variables change this behavior; each accepts `ON` or `OFF` and is read in the application process, including when XServer runs the policy:

| Name                               | Type   | Default Value | Description                                                                                      |
| ---------------------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------------ |
| XSCHED_SCHEDULER_SUSPEND_SYNC_HWQ  | string | OFF           | Synchronize the native hardware queue after pausing command launch.                              |
| XSCHED_SCHEDULER_SUSPEND_WAIT_ALL  | string | OFF           | Wait all commands when scheduling.     |
| XSCHED_SCHEDULER_SUSPEND_WAIT_IDLE | string | OFF           | Wait XQueues to become idle when scheduling. |

## Policy configuration

Set `XSCHED_POLICY` to a valid policy name to create a local scheduler with that policy.
If it is set, `XSCHED_SCHEDULER` is ignored in that process.
XServer also accepts a policy name as its command-line argument.

| Value | Full Name                            |
| ----- | ------------------------------------ |
| HPF   | Highest Priority First               |
| HHPF  | Heterogeneous Highest Priority First |
| CHPF  | CPU Highest Priority First           |
| UP    | Utilization Partition                |
| PUP   | Process Utilization Partition        |
| SPUP  | Strict Process Utilization Partition |
| KEDF  | K-Earliest Deadline First            |
| LAX   | Laxity-based                         |
| AWF   | Active Window First                  |
| CFS   | Completely Fair Scheduler            |
| MLFQ  | Multi-Level Feedback Queue           |

Memory scheduling uses the execution order and timeslices published by the selected policy.
See the [policy guide](../sched/README.md#memory-scheduling) for the current integration and instructions for adapting other policies.

## Memory Context Types

`XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT` selects how a queue determines the memory needed by upcoming work.
The default, `ALLOCATION`, includes all tracked allocations.
Prediction modes narrow this set to the regions expected to be accessed during an upcoming timeslice.
Applications using the APIs explicitly select the corresponding enum with `XQueueSetMemoryContextType()`:

| Value               | Enum                              | Behavior                                                               |
| ------------------- | --------------------------------- | ---------------------------------------------------------------------- |
| NONE                | `kMemoryContextNone`              | Disable memory scheduling                                            |
| ALLOCATION          | `kMemoryContextAllocation`        | Include all allocations tracked by the queue's memory manager          |
| BUFFER              | `kMemoryContextBuffer`            | Predict access from pointer arguments to the ends of their allocations |
| FAULT               | `kMemoryContextFault`             | Predict kernel accesses from page fault history                        |
| TEMPLATE            | `kMemoryContextTemplate`          | Predict accesses using offline templates and launch arguments          |
| TEMPLATE_AND_BUFFER | `kMemoryContextTemplateAndBuffer` | Use templates, with buffer prediction when no template matches         |
| TEMPLATE_AND_FAULT  | `kMemoryContextTemplateAndFault`  | Combine templates with page fault history for each kernel              |

For kernels, the selected predictor derives accesses from launch arguments, templates, or page fault history; memcpy and memset regions come directly from their API arguments.
Command latency estimates then identify which accesses belong to the requested timeslice.
See the [CUDA guide](../platforms/cuda/README.md#memory-access-prediction) for how these predictions are constructed.

### Prediction Files

Prediction uses access templates to identify memory regions and latency profiles to map commands to execution periods.
The first two variables select these input files at runtime; the remaining variables control sample collection when generating a latency profile:

| Variable                              | Default               | Description                                                     |
| ------------------------------------- | --------------------- | --------------------------------------------------------------- |
| `XSCHED_MEMORY_CONTEXT_TEMPLATE_FILE` | `memory_context.json` | Access templates matched by kernel symbol and launch dimensions |
| `XSCHED_EXECUTION_TIME_FILE`          | `execution_time.json` | Command latency profile; also the profiler's output path        |
| `XSCHED_EXECUTION_TIME_WARMUP_COUNT`  | 1000                  | Warmup samples discarded per profile entry                      |
| `XSCHED_EXECUTION_TIME_PROFILE_COUNT` | 5000                  | Samples required per entry, including warmup                    |

Both file paths are resolved relative to the application's working directory unless an absolute path is supplied.
The runtime reads each latency entry's `median` value in **nanoseconds**.
When generating this profile, set the profile count above the warmup count so that samples remain after warmup is discarded.
See [Profiling](../platforms/cuda/README.md#profiling) for the collection commands.

### Page Fault Prediction

The predictor learns accessed regions from observed page faults, but a region that has been prefetched may be reused without triggering further page faults.
The predictor therefore retains learned regions for a limited lifetime, expires old predictions, and extends the retry window when page faults recur.
The following parameters control this aging mechanism and accept integers from 1 to 1000000:

| Variable                              | Default | Description                                                 |
| ------------------------------------- | ------- | ----------------------------------------------------------- |
| `XSCHED_FAULT_TOKEN_INITIAL_CURRENT`  | 4       | Initial prediction lifetime                                 |
| `XSCHED_FAULT_TOKEN_INITIAL_NEXT`     | 8       | Initial retry window after that lifetime                    |
| `XSCHED_FAULT_TOKEN_RETRY_MULTIPLIER` | 8       | Retry window multiplier when page faults recur              |

In the current implementation, lifetimes are measured in feedback generations: each predictor update advances a generation, including updates with no observed page faults.
These values therefore describe how many feedback updates a prediction survives, rather than elapsed time or a count of kernel invocations.
See [Page Fault-Driven Prediction](../platforms/cuda/README.md#page-fault-driven-prediction) for how the runtime collects these updates.
