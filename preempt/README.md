# XPreempt

The XPreempt library mainly includes following parts:

- `HwCommand & HwQueue`: Base classes abstraction for hardware command and queue. They are specifically implemented in `platforms` for different hardware.
- `XQueue`: The main abstraction for task controlling and scheduling.
- `SchedAgent`: An agent that watches the status of XQueues and handles scheduling events.
- `XMemoryManager`: Tracks allocations and applies memory advice through the platform adapter.

## HwCommand & HwQueue

`HwCommand` is an encapsulation of driver API calling. We use this kind of abstraction to buffer API calling in XQueue, and use scheduler to decide when to launch it to `HwQueue`.

`HwQueue` is an encapsulation of hardware queue, e.g., `cudaStream` in CUDA, `hipStream` in ROCm, `ze_command_queue` in LevelZero, etc. When a `HwCommand` is launched to `HwQueue`, it will be executed in the hardware queue by calling respective driver API.

## XQueue

`XQueue` is a queue of XPU commands that are executed sequentially in the order of submission. `XQueue` is one-to-one mapped to a `HwQueue`.

### Memory Context

Memory scheduling needs to know which regions a queue will access during its next timeslice.
NVXSched represents this working set as a memory context, built from the access and execution-time predictions attached to commands when they are submitted.
XQueue walks the commands it still tracks in submission order, accumulates their estimated execution times, and combines the accesses that fall within the requested period.

This traversal covers both the launch worker's command log and the pending buffer, counting each command once.
The requested period is expressed in nanoseconds relative to the start of that sequence.
Commands crossing a period boundary contribute their full access sets, so memory preparation covers the whole command even when it spans timeslices.
Commands without a valid latency estimate do not advance the estimated time, which reduces the accuracy of the working set boundary.

Before issuing advice, XQueue intersects predicted accesses with live allocations recorded by its memory manager.
Applications can also choose `ALLOCATION` to prepare all tracked allocations or `NONE` to disable memory advice.
See [Memory Context Types](../protocol/README.md#memory-context-types) for the available prediction methods.

### XMemoryManager

`XMemoryManager` connects memory contexts to the platform's memory management operations.
It tracks allocations and delegates memory advice and feedback collection to `HwMemoryManager`.
CUDA queues in the same context share this manager, so `ALLOCATION` advice covers all tracked allocations in that CUDA context.

For prediction based on page faults, the manager also supplies the access history used to predict later commands.
After the scheduler's executor suspends a queue, it requests a feedback update through the manager before reporting operation completion.
Queues sharing a manager reuse the same update within that operation.
The updated history is used for subsequent submissions; commands already buffered retain their existing predictions.

## SchedAgent

When process starts, it will load `XPreempt` lib and initilize a `SchedAgent` with corresponding `Executor` and `Scheduler`.

- `Executor` implements `Execute` function to handle `Operation` that `SchedAgent` receives.
- `Scheduler` is an instance of `sched::Scheduler`. It will bind its `executor_` with `Executor`.

When `SchedAgent` receives `Event` from `XQueue`, it will call `RecvEvent` of `Scheduler` to handle.

The default mode, `APP`, leaves queue control to the application.
Setting `XSCHED_POLICY` selects a local policy, while `XSCHED_SCHEDULER=GLB` connects to XServer when no local policy is selected.
In global mode, the agent sends events through IPC and applies operations returned by the server.
Completion events let the server order dependent actions, such as suspending an outgoing task before preparing memory for the incoming one.
See the [scheduling module](../sched/README.md) for the event and operation protocol.
