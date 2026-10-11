#pragma once

#include <map>
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_map>

#include "xsched/utils/region.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cupti.h"
#include "xsched/cuda/hal/profile/protocol.h"

namespace xsched::cuda
{

class AccessRelation; // forward declaration

class Profiler
{
public:
    Profiler();
    ~Profiler();

    static bool GetKernelMemAccess(
        utils::Region &reads, utils::Region &writes, CUcontext ctx,
        CUfunction func,      uint64_t pgdim,      uint32_t pbdim,      void **params,
        CUfunction func_prev, uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev)
    {
        return g_profiler.GetKernelMemAccessInternal(reads, writes,
            ctx, func, pgdim, pbdim, params, func_prev, pgdim_prev, pbdim_prev, params_prev);
    }

    static int64_t GetKernelExecutionTimeNs(CUcontext ctx, CUfunction func,
                                            int32_t gdx, int32_t gdy, int32_t gdz,
                                            int32_t bdx, int32_t bdy, int32_t bdz)
    {
        return g_profiler.GetKernelExecutionTimeNsInternal(ctx, func, gdx, gdy, gdz, bdx, bdy, bdz);
    }

    static int64_t GetMemcpyExecutionTimeNs(CUpti_ActivityMemcpyKind kind, size_t size)
    {
        return g_profiler.GetMemcpyExecutionTimeNsInternal(kind, size);
    }

    static int64_t GetMemsetExecutionTimeNs(CUpti_ActivityMemoryKind kind, size_t size)
    {
        return g_profiler.GetMemsetExecutionTimeNsInternal(kind, size);
    }

private:
    bool GetKernelMemAccessInternal(
        utils::Region &reads, utils::Region &writes, CUcontext ctx,
        CUfunction func,      uint64_t pgdim,      uint32_t pbdim,      void **params,
        CUfunction func_prev, uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev) const;
    int64_t GetKernelExecutionTimeNsInternal(CUcontext ctx, CUfunction func,
                                             int32_t gdx, int32_t gdy, int32_t gdz,
                                             int32_t bdx, int32_t bdy, int32_t bdz) const;
    int64_t GetMemcpyExecutionTimeNsInternal(CUpti_ActivityMemcpyKind kind, size_t size) const;
    int64_t GetMemsetExecutionTimeNsInternal(CUpti_ActivityMemoryKind kind, size_t size) const;

    // kernels_[kernel][gdim][bdim] = execution time, read-only
    std::unordered_map<uint64_t, std::map<uint64_t, std::map<uint32_t, int64_t>>> kernel_;
    // relations_[kernel][gdim][bdim] = {read_relation[], write_relation[]}, read-only
    std::unordered_map<uint64_t, std::map<uint64_t, std::map<uint32_t, std::pair<
        std::vector<std::unique_ptr<AccessRelation>>,
        std::vector<std::unique_ptr<AccessRelation>>>>>> relations_;

    // memcpy_[kind][size] = execution time, read-only
    std::map<CUpti_ActivityMemcpyKind, std::map<size_t, int64_t>> memcpy_;

    // memset_[kind][size] = execution time, read-only
    std::map<CUpti_ActivityMemoryKind, std::map<size_t, int64_t>> memset_;

    static Profiler g_profiler;
};

} // namespace xsched::cuda
