#pragma once

#include <vector>

#include "xsched/preempt/xmem/xmem.h"
#include "xsched/preempt/hal/hw_mem.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cuda_ioctl.h"

namespace xsched::cuda
{

class CudaMemoryRequest : public preempt::HwMemoryRequest
{
public:
    CudaMemoryRequest(size_t size, uint32_t flags = CU_MEM_ATTACH_GLOBAL)
        : HwMemoryRequest(size), kFlags(flags) {}
    virtual ~CudaMemoryRequest() = default;

    uint32_t Flags() const { return kFlags; }

private:
    const uint32_t kFlags = 0;
};

class CudaMemoryManager : public preempt::HwMemoryManager
{
public:
    CudaMemoryManager(CUcontext ctx);
    virtual ~CudaMemoryManager() = default;

    /// @brief Get the XMemoryManager of the current CUDA context.
    /// @return The XMemoryManager of the current CUDA context.
    static std::shared_ptr<preempt::XMemoryManager> GetCurrentXMemoryManager();

    /// @brief Get the XMemoryManager of a CUDA context.
    /// @param ctx The CUDA context.
    /// @return The XMemoryManager of the specified CUDA context.
    static std::shared_ptr<preempt::XMemoryManager> GetXMemoryManager(CUcontext ctx);

    virtual void *AllocatePagable(std::shared_ptr<preempt::HwMemoryRequest> req) override;
    virtual void Free(void *ptr) override;
    virtual void Advise(std::shared_ptr<utils::Region> mem, XMemoryAdviseType advise) override;
    virtual void UpdateMemoryInfo() override;
    virtual void OnContextTypeChange(XMemoryContextType type) override;

private:
    void Hint(uint64_t begin, uint64_t end);
    void PrefetchH2D(uint64_t begin, uint64_t end, bool sync);

    struct PageFaultRange
    {
        uint64_t begin;
        uint64_t end;
        uint64_t last_fault_service_timestamp_ns;
    };

    bool QueryPageFaultRecords(uint64_t since_timestamp_ns, std::vector<PageFaultRange> &records);

    int uvm_fd_ = -1;
    CUcontext context_;
    NvProcessorUuid cpu_uuid_;
    NvProcessorUuid gpu_uuid_;

    static std::mutex g_cuda_xmm_mtx_;
    static std::map<CUcontext, std::shared_ptr<preempt::XMemoryManager>> g_cuda_xmms_;

};

} // namespace xsched::cuda
