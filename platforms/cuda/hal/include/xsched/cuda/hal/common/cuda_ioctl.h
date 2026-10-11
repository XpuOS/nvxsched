#pragma once

#include <cstddef>

namespace xsched::cuda
{

typedef unsigned char NvU8;
typedef unsigned long long NvU64;
typedef unsigned int NvU32;
typedef signed int NvS32;
typedef unsigned int NvU32;
typedef unsigned int NvU32;
typedef NvU32 NV_STATUS;

#define NV_UUID_LEN 16
typedef struct nv_uuid
{
    NvU8 uuid[NV_UUID_LEN];
} NvUuid;

typedef NvUuid NvProcessorUuid;

#define NV_UUID_HI(pUuid) (*((NvU64*)((pUuid)->uuid + (NV_UUID_LEN >> 1))))
#define NV_UUID_LO(pUuid) (*((NvU64*)((pUuid)->uuid + 0)))
#define CU_UUID_HI(pUuid) (*((NvU64*)((pUuid)->bytes + (NV_UUID_LEN >> 1))))
#define CU_UUID_LO(pUuid) (*((NvU64*)((pUuid)->bytes + 0)))

#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))
#define UVM_IOCTL_BASE(i) i

#define UVM_MIGRATE     UVM_IOCTL_BASE(51)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NvU32           flags;                                // IN
    NvU64           semaphoreAddress   NV_ALIGN_BYTES(8); // IN
    NvU32           semaphorePayload;                     // IN
    NvS32           cpuNumaNode;                          // IN
    NvU64           userSpaceStart     NV_ALIGN_BYTES(8); // OUT
    NvU64           userSpaceLength    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_PARAMS;

#define UVM_HINT        UVM_IOCTL_BASE(2046)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NvU64           unloaded_chunks;                      // OUT
    NV_STATUS       rmStatus;                             // OUT
} UVM_HINT_PARAMS;

} // namespace xsched::cuda
