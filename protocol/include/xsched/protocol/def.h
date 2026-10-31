#pragma once

#define XSCHED_DEFAULT_PREEMPT_LEVEL              kPreemptLevelBlock
#define XSCHED_DEFAULT_MEMORY_CONTEXT_TYPE        kMemoryContextAllocation
#define XSCHED_DEFAULT_COMMAND_THRESHOLD          16
#define XSCHED_DEFAULT_COMMAND_BATCH_SIZE         8

#define XSCHED_AUTO_XQUEUE_ENV_NAME                "XSCHED_AUTO_XQUEUE"                // = str[ON/OFF]      , default = OFF
#define XSCHED_AUTO_XQUEUE_LEVEL_ENV_NAME          "XSCHED_AUTO_XQUEUE_LEVEL"          // = int[1, 3]        , default = 1
#define XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT_ENV_NAME "XSCHED_AUTO_XQUEUE_MEMORY_CONTEXT" // = str[NONE/ALLOCATION/BUFFER/FAULT/TEMPLATE/TEMPLATE_AND_BUFFER/TEMPLATE_AND_FAULT], default = ALLOCATION
#define XSCHED_AUTO_XQUEUE_THRESHOLD_ENV_NAME      "XSCHED_AUTO_XQUEUE_THRESHOLD"      // = int[1, max_int64], default = 16
#define XSCHED_AUTO_XQUEUE_BATCH_SIZE_ENV_NAME     "XSCHED_AUTO_XQUEUE_BATCH_SIZE"     // = int[1, threshold], default = 8
#define XSCHED_AUTO_XQUEUE_PRIORITY_ENV_NAME       "XSCHED_AUTO_XQUEUE_PRIORITY"       // = int[-256, 255]   , default = 0
#define XSCHED_AUTO_XQUEUE_UTILIZATION_ENV_NAME    "XSCHED_AUTO_XQUEUE_UTILIZATION"    // = int[0, 100]
#define XSCHED_AUTO_XQUEUE_TIMESLICE_ENV_NAME      "XSCHED_AUTO_XQUEUE_TIMESLICE"      // = int[100, 100000]
#define XSCHED_AUTO_XQUEUE_DEADLINE_ENV_NAME       "XSCHED_AUTO_XQUEUE_DEADLINE"       // = int[1, max_int64]
#define XSCHED_AUTO_XQUEUE_KDEADLINE_ENV_NAME      "XSCHED_AUTO_XQUEUE_KDEADLINE"      // = int[1, max_int64]
#define XSCHED_AUTO_XQUEUE_LAXITY_ENV_NAME         "XSCHED_AUTO_XQUEUE_LAXITY"         // = int[1, max_int64]

#define XSCHED_MEMORY_CONTEXT_NAME_NONE                "NONE" // no memory context
#define XSCHED_MEMORY_CONTEXT_NAME_ALLOCATION          "ALLOCATION" // all allocated memory
#define XSCHED_MEMORY_CONTEXT_NAME_BUFFER              "BUFFER" // buffers pointed by command arguments
#define XSCHED_MEMORY_CONTEXT_NAME_FAULT               "FAULT" // fault-driven prediction only
#define XSCHED_MEMORY_CONTEXT_NAME_TEMPLATE            "TEMPLATE" // parameter-access mappings
#define XSCHED_MEMORY_CONTEXT_NAME_TEMPLATE_AND_BUFFER "TEMPLATE_AND_BUFFER" // buffer if no mapping is found
#define XSCHED_MEMORY_CONTEXT_NAME_TEMPLATE_AND_FAULT  "TEMPLATE_AND_FAULT" // template supplemented by faults
#define XSCHED_MEMORY_CONTEXT_TEMPLATE_FILE_ENV_NAME   "XSCHED_MEMORY_CONTEXT_TEMPLATE_FILE" // = str, default = memory_context.json

#define XSCHED_EXECUTION_TIME_FILE_ENV_NAME          "XSCHED_EXECUTION_TIME_FILE" // = str, default = execution_time.json
#define XSCHED_EXECUTION_TIME_WARMUP_COUNT_ENV_NAME  "XSCHED_EXECUTION_TIME_WARMUP_COUNT"  // = int[0, max_int64], default = 1000, warmup count
#define XSCHED_EXECUTION_TIME_PROFILE_COUNT_ENV_NAME "XSCHED_EXECUTION_TIME_PROFILE_COUNT" // = int[1, max_int64], default = 5000, profile count (include warmup)

#define XSCHED_ASCEND_LIB_ENV_NAME     "XSCHED_ASCEND_LIB"
#define XSCHED_COREX_LIB_ENV_NAME      "XSCHED_COREX_LIB"
#define XSCHED_COREX_RT_LIB_ENV_NAME   "XSCHED_COREX_RT_LIB"
#define XSCHED_CUDA_LIB_ENV_NAME       "XSCHED_CUDA_LIB"
#define XSCHED_CUPTI_LIB_ENV_NAME      "XSCHED_CUPTI_LIB"
#define XSCHED_CUDART_LIB_ENV_NAME     "XSCHED_CUDART_LIB"
#define XSCHED_CUDLA_LIB_ENV_NAME      "XSCHED_CUDLA_LIB"
#define XSCHED_HIP_LIB_ENV_NAME        "XSCHED_HIP_LIB"
#define XSCHED_LEVELZERO_LIB_ENV_NAME  "XSCHED_LEVELZERO_LIB"
#define XSCHED_OPENCL_LIB_ENV_NAME     "XSCHED_OPENCL_LIB"
#define XSCHED_VPI_LIB_ENV_NAME        "XSCHED_VPI_LIB"
// NEW_PLATFORM: New platform lib env names go here.

#define XSCHED_CUDA_UVM_ENV_NAME                       "XSCHED_CUDA_UVM"                       // = str[ON/OFF], default = OFF, if set to ON, preempt level should be set to Level-1
#define XSCHED_CUDA_PTDS_AUTO_XQUEUE_ENV_NAME          "XSCHED_CUDA_PTDS_AUTO_XQUEUE"          // = str[ON/OFF], default = ON, enable auto XQueue create for PTDS, effective when XSCHED_AUTO_XQUEUE is ON, not effective for single stream
#define XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS_ENV_NAME "XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS" // = str[ON/OFF], default = OFF, force the process to use one single cuda stream
#define XSCHED_CUDA_SINGLE_PTDS_PER_PROCESS_ENV_NAME   "XSCHED_CUDA_SINGLE_PTDS_PER_PROCESS"   // = str[ON/OFF], default = OFF, use single stream per process for PTDS, effective when XSCHED_CUDA_SINGLE_STREAM_PER_PROCESS is ON
#define XSCHED_CUDA_LV3_IMPL_ENV_NAME                  "XSCHED_CUDA_LV3_IMPL"                  // = str[TSG/TRAP], default = TRAP
#define XSCHED_LEVELZERO_SLICE_CNT_ENV_NAME            "XSCHED_LEVELZERO_SLICE_CNT"

// For XSched server, the channel key is 1, which is often reserved for system process.
// For clients, the channel key is their process ID.
#define XSCHED_SERVER_CHANNEL_KEY    1
#define XSCHED_SERVER_DEFAULT_PORT   50000

#define XSCHED_X11_MONITOR_DEFAULT_PORT     50001
#define XSCHED_WAYLAND_MONITOR_DEFAULT_PORT 50002

#define XSCHED_UNKNOWN_NAME        "Unknown"

// Set XSched scheduler type, default scheduler is APP.
// e.g., export XSCHED_SCHEDULER=GLB
#define XSCHED_SCHEDULER_ENV_NAME  "XSCHED_SCHEDULER"
#define XSCHED_SCHEDULER_NAME_APP  "APP" // Application Managed
#define XSCHED_SCHEDULER_NAME_LCL  "LCL" // Local Scheduler
#define XSCHED_SCHEDULER_NAME_GLB  "GLB" // Global Scheduler

// XSched scheduler XQueue suspending behaviors.
#define XSCHED_SCHEDULER_SUSPEND_SYNC_HWQ_ENV_NAME  "XSCHED_SCHEDULER_SUSPEND_SYNC_HWQ"  // = str[ON/OFF], default = OFF
#define XSCHED_SCHEDULER_SUSPEND_WAIT_ALL_ENV_NAME  "XSCHED_SCHEDULER_SUSPEND_WAIT_ALL"  // = str[ON/OFF], default = OFF
#define XSCHED_SCHEDULER_SUSPEND_WAIT_IDLE_ENV_NAME "XSCHED_SCHEDULER_SUSPEND_WAIT_IDLE" // = str[ON/OFF], default = OFF

// Set XSched policy type, e.g., export XSCHED_POLICY=HPF
// If set, scheduler type will be set to LCL and XSCHED_SCHEDULER will be ignored.
#define XSCHED_POLICY_ENV_NAME  "XSCHED_POLICY" // e.g., export XSCHED_POLICY=HPF
#define XSCHED_POLICY_NAME_HPF  "HPF"  // Highest Priority First
#define XSCHED_POLICY_NAME_HHPF "HHPF" // Heterogeneous Highest Priority First
#define XSCHED_POLICY_NAME_CHPF "CHPF" // CPU Highest Priority First
#define XSCHED_POLICY_NAME_UP   "UP"   // Utilization Partition
#define XSCHED_POLICY_NAME_PUP  "PUP"  // Process Utilization Partition
#define XSCHED_POLICY_NAME_SPUP "SPUP" // Strict Process Utilization Partition
#define XSCHED_POLICY_NAME_KEDF "KEDF" // K-Earliest Deadline First
#define XSCHED_POLICY_NAME_LAX  "LAX"  // Laxity-based
#define XSCHED_POLICY_NAME_AWF  "AWF"  // Active Window First
#define XSCHED_POLICY_NAME_CFS  "CFS"  // Complete Fair Scheduler
#define XSCHED_POLICY_NAME_MLFQ "MLFQ" // Multi-Level Feedback Queue
// NEW_POLICY: New policy type names go here.
