# Use NVXSched Explicitly

Applications can use NVXSched's APIs to control XQueues, provide scheduling hints, and prepare GPU memory explicitly.
These interfaces retain XSched's public header and function names.

## Example

*Using example refer to [examples/2_give_hints](../examples/Linux/2_give_hints/README.md)*

## API List

The APIs in [xqueue.h](./xsched/xqueue.h) create and configure XQueues, submit commands, wait for completion, and suspend or resume execution.
Applications can either manage XQueues themselves or use the hints in [hint.h](./xsched/hint.h) to configure a scheduling policy.
The associated flags, enum definitions, and result codes are declared in [types.h](./xsched/types.h).

## Memory Scheduling

An XQueue's memory context describes the regions needed by its upcoming work.
CUDA memory scheduling uses this context to prepare managed allocations through the modified UVM driver.
Follow the [CUDA guide](../platforms/cuda/README.md) to set up the driver and prediction files before using the APIs below.

The default memory context includes all allocations tracked by the queue's memory manager.
To prepare only the predicted working set, select a [prediction method](../protocol/README.md#memory-context-types) after creating the XQueue and before submitting commands.
For example, the following setting combines access templates with page fault history:

```cpp
XQueueSetMemoryContextType(xqueue, kMemoryContextTemplateAndFault);
```

The scheduler uses these contexts to prepare memory at context switches according to the execution plan published by the policy.
Applications can also issue advice explicitly with `XQueueAdviseMemory()`, specifying the execution period whose memory should be prepared.
For example, this call requests that the regions needed for the first 5 ms of the tracked command sequence be loaded onto the GPU:

```cpp
// Prepare the first 5 ms of the tracked command sequence.
XQueueAdviseMemory(xqueue, kMemoryAdviseLoad, 0, 5000000);
```

The two endpoints are relative offsets in **nanoseconds**.
XQueue uses predicted command latencies to identify the accesses within that period, then applies the requested advice to those regions.
With the default `kMemoryContextAllocation` setting, advice covers all tracked allocations regardless of the period.
CUDA supports the following advice types:

| Advice                  | CUDA behavior                                                 |
| ----------------------- | ------------------------------------------------------------- |
| `kMemoryAdviseWillNeed` | Update eviction priority for the selected regions             |
| `kMemoryAdviseLoad`     | Update priority and request asynchronous migration to the GPU |
| `kMemoryAdviseUnload`   | Currently unused                                              |

To integrate memory preparation into a custom scheduling policy, follow the [policy guide](../sched/README.md#how-to-implement-your-own-policy), which explains how the scheduler coordinates advice with suspension, resumption, and page fault feedback.
