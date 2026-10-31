# CUDA Platform Support for NVXSched

The CUDA adapter provides transparent scheduling, GPU preemption, and memory scheduling for NVIDIA GPUs.
It consists of a shim (`libshimcuda.so`), a hardware abstraction layer (`libhalcuda.so`), and an execution-time profiler (`libprofilecuda.so`).
Memory scheduling currently targets Linux and requires the modified NVIDIA UVM driver described in [Build and Install](../../README.md#build-and-install).

## Supported preemption level

The CUDA adapter supports XSched's three preemption levels, with backend availability determined by the GPU architecture and driver.
The table below lists the architecture-specific implementations in [arch.cpp](hal/src/arch/arch.cpp): Level-2 uses kernel instrumentation, and Level-3 uses traps to interrupt kernels.
The adapter also provides TSG-based Level-3 preemption through the driver; see the [CUDA configuration](../../protocol/README.md#cuda-xqueue-specific-configuration) for backend selection.

<table>
  <tr>
    <th align="center">Platform</th>
    <th align="center">XPU</th>
    <th align="center">Shim</th>
    <th align="center">Level-1</th>
    <th align="center">Level-2</th>
    <th align="center">Level-3</th>
  </tr>
  <tr>
    <td align="center" rowspan="4">CUDA</td>
    <td align="center">NVIDIA Ampere GPUs (sm86)</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center" rowspan="4">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Volta GPUs (sm70)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td align="center">NVIDIA Kepler GPUs (sm35)</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">Other NVIDIA GPUs</td>
    <td align="center">🔘</td>
    <td align="center">🔘</td>
  </tr>
</table>

## Usage

Build the CUDA platform adapter:

```bash
make cuda
```

For transparent scheduling, see the [Linux transparent scheduling example](../../examples/Linux/1_transparent_sched/README.md).

The following commands use the default installation directory, `output/`, and run from the project root.
The build creates `libcuda.so` and `libcuda.so.1` symlinks to the shim, so adding `output/lib` to `LD_LIBRARY_PATH` enables CUDA interception.
For a custom installation, substitute its path for `output`.

### Run with Compute Scheduling

Start the scheduling service in one terminal:

```bash
./output/bin/xserver HPF 50000
```

In an application terminal, enable automatic XQueues and connect to the service:

```bash
export LD_LIBRARY_PATH="$PWD/output/lib:$LD_LIBRARY_PATH"
unset XSCHED_POLICY
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_CUDA_UVM=ON
export XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT=NONE
export XSCHED_AUTO_XQUEUE_PRIORITY=1

/path/to/application [arguments]
```

With `HPF`, the highest-priority ready queues on each device may run while lower-priority queues are suspended.
To give this application precedence over another, run the second application with a lower priority, such as `XSCHED_AUTO_XQUEUE_PRIORITY=0`.
Queues with equal priority may run concurrently.
See the [policy list](../../sched/README.md#policies) to choose another scheduling objective.

### Run with Memory Scheduling

Memory scheduling combines predicted accesses with the execution schedule to prepare the working sets of upcoming tasks.
The example below uses `TEMPLATE_AND_FAULT`: access templates predict regions from kernel arguments, and runtime page fault history supplements accesses that the templates miss.
First prepare an access template file and an execution-time profile for each application as described in [Profiling](#profiling).
Then start the service. This example uses `SPUP` policy (strict process utilization partition), which currently implements the memory scheduling interfaces:

```bash
./output/bin/xserver SPUP 50000
```

Run each application in a separate terminal with its own prediction files:

```bash
export LD_LIBRARY_PATH="$PWD/output/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XSCHED_CUPTI_LIB=/path/to/libcupti.so
unset XSCHED_POLICY
export XSCHED_SCHEDULER=GLB
export XSCHED_AUTO_XQUEUE=ON
export XSCHED_AUTO_XQUEUE_LEVEL=1
export XSCHED_CUDA_UVM=ON
export XSCHED_AUTO_XQUEUE_TIMESLICE=50000
export XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT=TEMPLATE_AND_FAULT
export XSCHED_MEMORY_CONTEXT_TEMPLATE_FILE=/path/to/template.json
export XSCHED_EXECUTION_TIME_FILE=/path/to/execution_time.json

/path/to/application [arguments]
```

Replace the application, CUPTI, and prediction paths with those on your system.
With these settings, `SPUP` rotates processes with an average timeslice of 50 ms (`50000` microseconds).
It divides execution time according to process utilization weights, giving equal base timeslices to processes with equal weights.
Use [xcli](../../service/README.md#examples) to adjust these weights and the average timeslice.

The [memory context types](../../protocol/README.md#memory-context-types) also allow other preparation strategies, such as using templates alone, learning from page fault history alone, or preparing all tracked allocations.
The prediction methods and their input requirements are described below.

## Memory Access Prediction

To prepare a working set before execution, NVXSched predicts a command's memory accesses when it is submitted to an XQueue.
Memcpy and memset specify their regions directly in their arguments, while kernel accesses require prediction from arguments or execution history.
The following methods supply these kernel predictions.

### Template-Based Prediction

Templates describe the relationship between kernel arguments and accessed memory, covering fixed-size, linear-size, and strided regions.
At runtime, the predictor matches a kernel's full symbol and grid and block dimensions, then evaluates the matching templates using its launch arguments.
Some mappings also use arguments from the preceding kernel to resolve an access.
Because the template describes a relationship rather than a fixed address, it can predict accesses at different addresses across executions.

Select the template file with `XSCHED_MEMORY_CONTEXT_TEMPLATE_FILE`; [Template Mappings](#template-mappings) describes how to generate it, and the [mapping implementation](hal/src/predict/mapping.cpp) defines its formats.
For kernels or accesses not covered by templates, choose a combined mode: `TEMPLATE_AND_BUFFER` falls back to buffer prediction when no template matches, while `TEMPLATE_AND_FAULT` supplements template predictions with page fault history for the same kernel.

### Page Fault-Driven Prediction

Some kernel addresses depend on data produced during GPU execution and cannot be determined from launch arguments alone.
Page fault history reveals accesses to pages that were missing when earlier kernels ran.
After the scheduler's executor suspends a task, it asks the CUDA memory manager to synchronize the context, collect CUPTI kernel intervals, and query page fault records from the UVM driver.
The predictor associates page faults with kernels by timestamp and records the regions under each kernel symbol.
Subsequent commands use this history when they are submitted; already buffered commands retain their predictions.

Successfully prefetched pages may be reused without triggering further page faults, so a lack of new records does not by itself show that a learned region is no longer needed.
The predictor therefore ages its history, retaining learned regions for a limited lifetime and extending their lifetime when page faults recur.
The [page fault prediction settings](../../protocol/README.md#page-fault-prediction) control this behavior.

This timestamp-based attribution is approximate when kernels overlap, because one page fault can fall within multiple kernel intervals.
The current collection path assumes one GPU and CUDA context per process, and its synchronization and CUPTI tracing add overhead to suspension.
These constraints matter when choosing between prediction based on page faults and a template-only configuration.

### Buffer-Based Prediction

Buffer prediction finds pointers in kernel arguments and looks them up in the allocation map.
For each recognized pointer, it predicts access from that address to the end of the allocation.
This requires no access templates, but can overestimate accesses when frameworks suballocate tensors from large memory pools.

### CUDA Graphs

NVXSched treats a submitted CUDA Graph as one command for memory prediction.
It combines the predicted accesses of kernel, memcpy, and memset nodes into one context and estimates duration by summing their known latencies.
Because this estimate does not model branch overlap, it can overestimate the duration of parallel graphs.
The prediction path currently supports only these three node types.

## Profiling

Memory scheduling uses two kinds of offline data: execution latencies determine which commands fit within an upcoming timeslice, and access templates describe the memory those commands need.
Collect both from runs that match the application's kernel symbols, launch dimensions, and CUDA libraries so that the runtime can find the corresponding entries.

### Execution Latencies

The CUDA build installs a timing profiler in `output/lib/profile`.
To collect latencies, run the application separately with the profiler's library directory in place of the scheduling shim:

```bash
unset LD_PRELOAD
unset XSCHED_POLICY XSCHED_SCHEDULER XSCHED_AUTO_XQUEUE
export LD_LIBRARY_PATH="$PWD/output/lib/profile${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XSCHED_CUPTI_LIB=/path/to/libcupti.so
export XSCHED_EXECUTION_TIME_FILE="$PWD/execution_time.json"
export XSCHED_EXECUTION_TIME_WARMUP_COUNT=1000
export XSCHED_EXECUTION_TIME_PROFILE_COUNT=5000

/path/to/application [arguments]
```

The profiler records kernel, memcpy, and memset durations and writes a JSON profile containing entries with enough samples.
`PROFILE_COUNT` includes the samples discarded during warmup, so it must exceed `WARMUP_COUNT`.
Adjust both values to the application's repetition count; entries that do not reach the required count are omitted from the output.

At runtime, NVXSched matches kernels by symbol hash and launch dimensions, and matches memcpy and memset commands by operation kind and size.
It reads the matched entry's `median` latency in **nanoseconds** to estimate how much work fits in a timeslice.
An unmatched command has no valid time estimate, reducing the accuracy of that timeslice's working set prediction.

### Template Mappings

The [memory tracing tool](../../tools/trace/README.md) collects kernel memory accesses and launch arguments using NVBit, and the [template generation tool](../../tools/template/README.md) derives access templates from these traces.
Both tools will be open-sourced soon under `tools/`.

## Driver Interfaces

The CUDA shim redirects supported device allocations to managed memory so that their pages can move between GPU and host memory.
XPreempt intersects predicted accesses with tracked allocations, then passes memory advice to the CUDA memory manager for application through the modified UVM driver.
The driver exposes three interfaces for controlling placement and collecting feedback:

| Interface      | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| `UVM_HINT`     | Prioritize regions in the driver's eviction order              |
| `UVM_MIGRATE`  | Request migration of an incoming working set to the GPU        |
| `UVM_QUERY_PF` | Collect page fault regions and timestamps for prediction       |

At a context switch, `WillNeed` advice adjusts eviction priority for future working sets, and `Load` additionally requests migration of the incoming working set.
If migration needs more GPU space, the driver evicts resident pages according to the updated order; `Unload` has no separate action in the current CUDA implementation.
Migration is asynchronous, so completing a memory operation means the advice has been issued, rather than that every transfer has finished.

The feedback query completes this cycle by reporting page faults from accesses that prediction missed, allowing later commands to use the updated history.
See [cuda_ioctl.h](hal/include/xsched/cuda/hal/common/cuda_ioctl.h) for interface definitions and [Build and Install](../../README.md#build-and-install) for the matching driver.
