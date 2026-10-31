#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <climits>

#include "xsched/utils/hash.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

typedef uint64_t KernelNameHash;
typedef uint64_t GridDim;
typedef uint32_t BlockDim;

struct KernelPredictionInfo
{
    CUfunction func;
    KernelNameHash hash;
    GridDim gdim;
    BlockDim bdim;
    void **params;
    std::vector<size_t> param_sizes;
};

KernelNameHash GetKernelNameHash(CUcontext ctx, CUfunction func);

inline GridDim PackGridDim(int32_t gdx, int32_t gdy, int32_t gdz)
{
    XASSERT(0 <= gdx && gdx <= INT_MAX        , "invalid gdx %d", gdx);
    XASSERT(0 <= gdy && gdy <= ((1 << 16) - 1), "invalid gdy %d", gdy);
    XASSERT(0 <= gdz && gdz <= ((1 << 16) - 1), "invalid gdz %d", gdz);
    return ((uint64_t)gdx << 32) + ((uint64_t)gdy << 16) + (uint64_t)gdz;
}

inline void UnpackGridDim(const GridDim gdim, int32_t &gdx, int32_t &gdy, int32_t &gdz)
{
    gdx = gdim >> 32;            // max = 2 ^ 31 - 1
    gdy = (gdim >> 16) & 0xFFFF; // max = 2 ^ 16 - 1
    gdz = gdim & 0xFFFF;         // max = 2 ^ 16 - 1
}

inline BlockDim PackBlockDim(int32_t bdx, int32_t bdy, int32_t bdz)
{
    // |---- 12 bit ----|---- 12 bit ----|---- 8 bit ----|
    XASSERT(0 <= bdx && bdx <= 1024, "invalid bdx %d", bdx);
    XASSERT(0 <= bdy && bdy <= 1024, "invalid bdy %d", bdy);
    XASSERT(0 <= bdz && bdz <= 64  , "invalid bdz %d", bdz);
    return ((uint32_t)bdx << 20) + ((uint32_t)bdy << 8) + (uint32_t)bdz;
}

inline void UnpackBlockDim(const BlockDim bdim, int32_t &bdx, int32_t &bdy, int32_t &bdz)
{
    bdx = bdim >> 20 & 0xFFF;  // max = 1024
    bdy = (bdim >> 8) & 0xFFF; // max = 1024
    bdz = bdim & 0xFF;         // max = 64
}

} // namespace xsched::cuda
