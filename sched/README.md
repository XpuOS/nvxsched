# Scheduling Module for NVXSched

## Schedulers

### Application Managed Scheduler

`AppManagedScheduler` is an empty scheduler.
Users need to call NVXSched APIs manually (refer to [NVXSched API](../include/README.md)) to control XQueues.
This is the default scheduler (`APP`).

### Local Scheduler

`local.cpp` implements processing logic of a scheduler. When running a `LocalScheduler`, it will start a thread running function `Worker`. `Worker` contains a loop to take events in `event_queue_` and call `Sched` API of policy. `local.cpp` also defines following function:

- `Suspend` and `Resume` are called by `policy` to record scheduling decisions, which the scheduler applies after `Sched` returns.
- `SetPolicy` is used to changes policy online

**Note:**

- **If a process's scheduler type is `LocalScheduler`, it means the process can only schedule xqueues created by itself.**
- **The scheduler type of `XServer` is `LocalScheduler`, with a global view of registered XQueues.**

### Global Scheduler

<img src="../docs/img/global-scheduler.png" alt="Global Scheduler Framework" width="600" />

The global scheduler uses two IPC channels.
Its `Worker` thread receives `Operation` messages through `recv_chan_` and calls the executor bound by `SetExecutor()` in [scheduler.h](../sched/include/xsched/sched/scheduler/scheduler.h).

- `send_chan_` uses key `XSCHED_SERVER_CHANNEL_KEY` and its type is sender. When `preempt::SchedAgent` call `RecvEvent` to force scheduler to receive events, `send_chan_` will send event to receiver and do nothing else.
- `recv_chan_` uses key bound to its PID and its type is receiver.

If a process wants to use `GlobalScheduler`, user must first run XServer. XServer will create a `LocalScheduler` and IPC channels to communicate with each process's scheduler, receiving events sended by process's `send_chan_` and generating operations by policy in its `LocalScheduler`, then send these operations back to process's `recv_chan_`.
Set `XSCHED_SCHEDULER=GLB` and leave `XSCHED_POLICY` unset in these applications, because a valid `XSCHED_POLICY` setting selects a local scheduler.

## Memory Scheduling

Preparing GPU memory requires both a working set prediction and a plan for when that working set will be used.
NVXSched therefore extends the policy interface with `SetTimeslice()`, which publishes a queue's execution budget in microseconds, and `UpdateRunQueue()`, which publishes the upcoming queue order.
Together, they let the scheduler prepare memory according to the policy's execution plan.

The [policy table](#policies) lists which policies currently implement these interfaces.

When the published run order changes, the scheduler coordinates a context switch as follows:

1. Suspends outgoing queues and waits for operation completion. Modes using page fault history update prediction feedback before acknowledging suspension.
2. Visits queues in reverse execution order and issues `WillNeed` advice for future queues. Updating eviction priority in this order protects pages needed sooner.
3. Issues `Load` advice for the incoming queues to update their page priority and request migration to the GPU.
4. Resumes the incoming queues after their memory advice has been issued.

The scheduler sends each queue a time interval rather than a list of pages.
The application process maps that interval to its tracked commands and constructs the memory context locally, keeping command and allocation metadata out of server communication.
See [XPreempt](../preempt/README.md#memory-context) for context construction and the [CUDA guide](../platforms/cuda/README.md#driver-interfaces) for how advice controls eviction and migration.

## Protocol

### Event

`event` are used in communication among `preempt::XQueue`, `preempt::SchedAgent`, `sched::Scheduler`, `service::Server`.

When XQueues' status change, e.g. become ready when new commands are submitted to the XQueue, become idle when all commands are completed, XQueue will send event to `SchedAgent`. `SchedAgent` will process these events by itself or sent them to the XServer according to the type of its scheduler.

### Operation

`operation` is used in communication among `service::Server`, `sched::Scheduler`, `preempt::SchedAgent`.

In global mode, when policy in `sched::Scheduler` (LocalScheduler) generates operations, it calls `Execute()` bound to `service::Server`. Then `Execute()` finds the corresponding application process and sends these operations to `preempt::SchedAgent` to suspend or resume XQueues, change queue configuration, or apply memory advice.

Completion acknowledgements let the scheduler order dependent actions, such as issuing memory advice after suspension.
They acknowledge the operation handler's completion; whether this includes waiting for hardware depends on the operation and its flags.
For example, CUDA `Load` advice requests asynchronous migration, so its acknowledgement does not mean every page has arrived.

### Hint

`hint` are customized data structures to change policy parameters. Developers can define their own hint types when implementing a new policy. The hint types we support now are as follows:

| Name                  | Used Policy         | Description                                                                                |
| --------------------- | ------------------- | ------------------------------------------------------------------------------------------ |
| kHintTypePriority     | `HPF`, `HHPF`       | Change priority of the xqueue                                                              |
| kHintTypeUtilization  | `UP`, `PUP`, `SPUP` | Set a queue weight for UP or a process weight for PUP/SPUP                                 |
| kHintTypeTimeslice    | `UP`, `PUP`, `SPUP` | Set the round budget for UP or the average process timeslice for PUP/SPUP, in microseconds |
| kHintTypeDeadline     | `KEDF`              | Set a deadline relative to the queue's ready time, in microseconds                         |
| kHintTypeKDeadline    | `KEDF`              | Set the number of queues allowed to run concurrently                                       |
| kHintTypeLaxity       | `LAX`               | Set the allowed delay and the priorities before and after it expires                       |
| kHintTypeWindowActive | `AWF`               | Change window activity                                                                     |
| kHintTypePriority     | `CFS`               | Set priority-derived CFS weights                                                           |

## Policies

`policy.cpp` constructs policy implementations, all of which derive from `sched::Policy`.

Now we have implemented several policies, including:

| Value | Full Name                            | Description                                                      | Memory Scheduling |
| ----- | ------------------------------------ | ---------------------------------------------------------------- | ----------------- |
| HPF   | Highest Priority First               | Run ready queues with the highest priority on each device        |                   |
| HHPF  | Heterogeneous Highest Priority First | Use the highest ready-queue priority across all devices          |                   |
| CHPF  | CPU Highest Priority First           | Read CPU scheduling priorities from /proc to assign GPU priority |                   |
| UP    | Utilization Partition                | Assign time slices with weights to each xqueue                   |                   |
| PUP   | Process Utilization Partition        | Assign time slices with weights to each process                  |                   |
| SPUP  | Strict Process Utilization Partition | Rotate processes with weighted timeslices                        | ✅                |
| KEDF  | K-Earliest Deadline First            | Firstly run the k xqueues with earliest deadline                 |                   |
| LAX   | Laxity-based                         | Switch between configured priorities when queue laxity expires   |                   |
| AWF   | Active Window First                  | Firstly run the xqueue in the active window                      |                   |
| CFS   | Completely Fair Scheduler            | Linux-like CFS using vruntime and weights                        |                   |
| MLFQ  | Multi-Level Feedback Queue           | Features Soft Priority Recovery for interactive tasks            |                   |

We are adapting more scheduling policies to support memory scheduling.

## How to Implement Your Own Policy

### 1. Create a new policy class

Create a new policy class in `sched/include/xsched/sched/policy` directory, and inherit from `sched::Policy`. Implement the following two functions:

- `Sched` is the main function of the policy, it will be called by `sched::Scheduler` to schedule xqueues.
- `RecvHint` is used to receive hints and change policy parameters.

To add memory scheduling, also call `SetTimeslice()` for each queue and `UpdateRunQueue()` with the upcoming execution order, placing incoming queues first.
The scheduler then performs the memory preparation sequence described above before applying the resumption decisions.
It triggers this preparation when the published queue order changes; publishing the same order does not repeat the advice.
See [spup.cpp](src/policy/spup.cpp) for a policy that combines these interfaces.

### 2. Register the new policy

1. Add the new policy type to the enum `XPolicyType` in `include/xsched/types.h`.
2. Add the new policy type to the function `CreatePolicy()` in `sched/src/policy/policy.cpp`.
3. Add the new policy type to the definition of `XSCHED_POLICY` in `protocol/include/xsched/protocol/def.h`.
4. Add the new policy type to the map `kPolicyNames` in `protocol/src/names.cpp`.
5. Add the new hint type by need in `hint.h` and `hint.cpp`.
