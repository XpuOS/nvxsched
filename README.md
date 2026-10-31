# NVXSched: Compute and Memory Scheduling for NVIDIA GPUs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)

## About

NVXSched brings flexible multitasking to NVIDIA GPUs.
Built on [XSched](docs/xsched-osdi25.pdf), it combines **preemptive compute scheduling** with **proactive memory scheduling**, so applications can share a GPU even when their combined memory demands exceed its capacity.
Existing CUDA applications run without source code changes, while users can choose scheduling policies to match their priorities and performance needs.

Compared with CUDA Unified Memory's demand paging, NVXSched proactively prepares the incoming task's working set at each context switch, rather than relying on costly page faults during execution.
It predicts upcoming memory accesses using queued GPU commands and the execution schedule, then uses these predictions to guide eviction and load the working set before the task resumes.

### Features

- **Transparency**: Works with existing applications without code changes.
- **Preemptive compute scheduling**: Preempts running GPU tasks and switches between them to improve responsiveness and fairness.
Supports diverse scheduling policies.
- **Memory oversubscription:** Share GPU memory across applications whose combined memory footprint exceeds device capacity.
- **Predictive memory management:** Combine offline access templates with runtime page fault feedback to predict memory aceesses and coordinate page migration with task execution.

## Getting Started

### Build and Install

Requirements:

- CMake
- A C++20 compiler (GCC recommended)
- NVIDIA driver package: kernel modules (`nvidia.ko`, `nvidia_uvm.ko`, `nvidia_modeset.ko`, `nvidia_drm.ko`) and CUDA driver (`libcuda.so`)
- CUPTI (`libcupti.so`)

Run the following commands from the project root:

```bash
git submodule update --init --recursive

# Build the CUDA HAL, shim, execution-time profiler, and scheduling service.
# The default installation directory is output/ under the project root.
make cuda

# install to a custom path
make cuda INSTALL_PATH=/path/to/install
```

Supported NVIDIA driver versions:

| Driver Version | KMD Branch |
| -------------- | ---------- |
| 580.95.05 | `msched-580.95.05` |

We are adding support for more driver versions.

Check the **driver version** with `nvidia-smi` and substitute it for `<KMD_VERSION>` below:

```bash
nvidia-smi
cd 3rdparty/nvidia-kmd
git checkout msched-<KMD_VERSION>
make -j"$(nproc)"
sudo make msched
cd ../..
```

### Use Cases

#### Transparently Schedule CUDA Applications

NVXSched is designed to be transparent to applications.
By loading the CUDA shim and setting a few [environment variables](protocol/README.md), you can schedule existing CUDA applications without modifying their source code.
For scheduling across processes, `xserver` coordinates the applications and `xcli` lets you monitor queues or adjust the policy through scheduling hints.
See the [CUDA usage guide](platforms/cuda/README.md#usage) and [service guide](service/README.md) for details.

#### Combined Compute and Memory Scheduling

When applications need more memory than the GPU can hold, NVXSched uses host memory to hold pages that are not needed immediately.
At each context switch, it predicts the incoming task's working set and uses the execution schedule to guide which pages to populated and evict.
See [memory scheduling](platforms/cuda/README.md#run-with-memory-scheduling) for configuration and [profiling](platforms/cuda/README.md#profiling) for preparing prediction data.

#### Customize Scheduling

Applications that need direct control can link with NVXSched to manage XQueues or provide hints to the scheduling policy.
See the [API guide](include/README.md) and the examples for [giving hints](examples/Linux/2_give_hints/README.md), [intra-process scheduling](examples/Linux/3_intra_process_sched/README.md), and [manual scheduling](examples/Linux/4_manual_sched/README.md).

To implement a new scheduling policy, see the [policy guide](sched/README.md#how-to-implement-your-own-policy).
The [example list](examples/README.md) includes additional XSched compute scheduling and application integration examples.

## Architecture and Workflow

NVXSched builds on XSched's preemptible command queue abstraction, XQueue, and extends it with predictions of upcoming memory accesses.
The scheduler uses these predictions together with its execution plan to coordinate compute and memory scheduling.
The CUDA shim, XPreempt, and CUDA HAL run in each application process, while `xserver` coordinates decisions across processes:

- **[CUDA shim](platforms/cuda):** intercepts CUDA API calls and redirects commands to XQueues, allowing existing applications to use NVXSched transparently. It also redirects device allocations to managed memory for oversubscription.
- **[XPreempt](preempt):** implements XQueue interfaces for command submission, waiting, suspension, and resumption, and constructs memory contexts. A memory context describes the regions predicted to be accessed by commands within a given execution period.
- **[CUDA HAL](platforms/cuda):** implements GPU preemption and predicts command execution times and memory accesses. It combines access templates with page fault feedback to account for both static and dynamic accesses.
- **[Scheduler](sched):** decides which XQueues may run according to the selected policy. For memory scheduling, the policy also publishes the upcoming execution order and timeslices to guide memory preparation.
- **[Memory manager](platforms/cuda/README.md#driver-interfaces):** works with the modified NVIDIA UVM driver to prioritize pages for upcoming tasks and migrate their working sets to the GPU.

When an application submits a command to an XQueue, NVXSched predicts its memory accesses and execution time.
At a context switch, the scheduler suspends outgoing tasks and uses the upcoming execution order and timeslices to identify working sets.
It then updates page eviction priorities and requests migration of the incoming working sets before resuming those tasks.
Page faults observed during execution feed back into predictions for subsequent commands, helping cover accesses that earlier predictions missed.

## Documentation

| Guide | Contents |
| ----- | -------- |
| [CUDA](platforms/cuda/README.md) | Running applications, profiling, and memory prediction |
| [XPreempt](preempt/README.md) | Command queues and memory contexts |
| [Scheduling](sched/README.md) | Scheduling policies and how to add a policy |
| [Configuration](protocol/README.md) | Environment variables and their defaults |
| [Public APIs](include/README.md) | Queue control, scheduling hints, and memory advice |
| [Service](service/README.md) | Using `xserver` and `xcli` |
| [Platform adapters](platforms/README.md) | Hardware abstraction and API interception |

The [XSched introduction](docs/xsched-intro-2025-en.md) and its [Chinese version](docs/xsched-intro-2025-zh.md) describe the underlying compute scheduling framework.

## Development Plan

We will continue to improve NVXSched's scheduling capabilities and support more NVIDIA GPUs and driver versions. Please stay tuned!

- [ ] Support spatial multiplexing to schedule GPU tasks running concurrently.
- [ ] Extend memory scheduling to more scheduling policies.
- [ ] Support more NVIDIA driver versions.

## Contributing

NVXSched is designed to be extensible and flexible.
We welcome contributions:

- Implement a new scheduling policy or extend an existing policy with memory scheduling. See the [guide](sched/README.md).
- Improve memory access prediction, GPU preemption, or driver support. See the [CUDA guide](platforms/cuda/README.md).
- Integrate NVXSched into GPU applications and inference systems.
- Report or fix issues, and improve documentation and examples.

## Citation

If you use NVXSched for your research, please cite our papers:

```bibtex
@misc{Shen2026msched,
  title={Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling},
  author={Weihang Shen and Yinqiu Chen and Rong Chen and Haibo Chen},
  year={2026},
  eprint={2512.24637},
  archivePrefix={arXiv},
  primaryClass={cs.OS},
  url={https://arxiv.org/abs/2512.24637},
}

@inproceedings{Shen2025xsched,
  title = {{XSched}: Preemptive Scheduling for Diverse {XPU}s},
  author = {Weihang Shen and Mingcong Han and Jialong Liu and Rong Chen and Haibo Chen},
  booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
  year = {2025},
  isbn = {978-1-939133-47-2},
  address = {Boston, MA},
  pages = {671--692},
  url = {https://www.usenix.org/conference/osdi25/presentation/shen-weihang},
  publisher = {USENIX Association},
  month = jul
}
```

## Contact Us

- For technical questions and feature requests, please submit via GitHub [Issues](https://github.com/XpuOS/nvxsched/issues).
- For collaborations and partnerships, please reach out at [rongchen@sjtu.edu.cn](mailto:rongchen@sjtu.edu.cn).
