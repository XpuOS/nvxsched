#include <sys/ioctl.h>

#include "cuxtra/cuxtra.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_mem.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

#define CUDA_UVM_PAGE_SIZE 4096

using namespace xsched::cuda;
using namespace xsched::preempt;

std::mutex CudaMemoryManager::g_cuda_xmm_mtx_;
std::map<CUcontext, std::shared_ptr<XMemoryManager>> CudaMemoryManager::g_cuda_xmms_;

std::shared_ptr<XMemoryManager> CudaMemoryManager::GetCurrentXMemoryManager()
{
    CUcontext current_ctx = nullptr;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    return GetXMemoryManager(current_ctx);
}

std::shared_ptr<XMemoryManager> CudaMemoryManager::GetXMemoryManager(CUcontext ctx)
{
    std::lock_guard<std::mutex> lock(g_cuda_xmm_mtx_);
    auto it = g_cuda_xmms_.find(ctx);
    if (it != g_cuda_xmms_.end()) return it->second;
    auto cuda_mm = std::make_shared<CudaMemoryManager>(ctx);
    auto cuda_xmm = std::make_shared<XMemoryManager>(cuda_mm);
    g_cuda_xmms_[ctx] = cuda_xmm;
    return cuda_xmm;
}

CudaMemoryManager::CudaMemoryManager(CUcontext ctx): context_(ctx)
{
    CUdevice dev;
    CUcontext current_ctx;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current_ctx));
    XASSERT(current_ctx == ctx,
            "create CudaMemoryManager failed: current context (%p) does not match context (%p)",
            current_ctx, ctx);
    CUDA_ASSERT(Driver::CtxGetDevice(&dev));
    CUDA_ASSERT(Driver::DeviceGetUuid((CUuuid*)&gpu_uuid_, dev));

    uvm_fd_ = cuXtraGetUvmVaSpaceFd();
    XINFO("found fd %d for /dev/nvidia-uvm", uvm_fd_);

    // cpu_uuid_ = 73772a14-2c41-4750-a27b-d4d74e0f5ea6
    // defined in 3rdparty/open-gpu-kernel-modules/kernel-open/nvidia-uvm/nvCpuUuid.c
    NV_UUID_HI(&cpu_uuid_) = 0x73772a142c414750;
    NV_UUID_LO(&cpu_uuid_) = 0xa27bd4d74e0f5ea6;
}

void *CudaMemoryManager::AllocatePagable(std::shared_ptr<HwMemoryRequest> req)
{
    auto cuda_req = std::dynamic_pointer_cast<CudaMemoryRequest>(req);
    XASSERT(cuda_req != nullptr, "memory request is not a CudaMemoryRequest");
    CUdeviceptr dptr = 0;
    CUDA_ASSERT(Driver::MemAllocManaged(&dptr, cuda_req->Size(), cuda_req->Flags()));
    return (void *)dptr;
}

void CudaMemoryManager::Free(void *ptr)
{
    CUDA_ASSERT(Driver::MemFree_v2((CUdeviceptr)ptr));
}

void CudaMemoryManager::Advise(std::shared_ptr<utils::Region> mem, XMemoryAdviseType advise)
{
    XDEBG("advise (%d) %zu MiB of memory", (int)advise, mem->Size() >> 20);
    switch (advise)
    {
    case kMemoryAdviseWillNeed:
    {
        mem->ForEach([&](uint64_t begin, uint64_t end) -> void {
            begin = ROUND_DOWN(begin, CUDA_UVM_PAGE_SIZE);
            end = ROUND_UP(end, CUDA_UVM_PAGE_SIZE);
            this->Hint(begin, end);
        });
        break;
    }
    case kMemoryAdviseLoad:
    {
        mem->ForEach([&](uint64_t begin, uint64_t end) -> void {
            begin = ROUND_DOWN(begin, CUDA_UVM_PAGE_SIZE);
            end = ROUND_UP(end, CUDA_UVM_PAGE_SIZE);
            this->Hint(begin, end);
        });
        mem->ForEach([&](uint64_t begin, uint64_t end) -> void {
            begin = ROUND_DOWN(begin, CUDA_UVM_PAGE_SIZE);
            end = ROUND_UP(end, CUDA_UVM_PAGE_SIZE);
            this->PrefetchH2D(begin, end, false);
        });
        break;
    }
    default:
        break;
    }
}

void CudaMemoryManager::Hint(uint64_t begin, uint64_t end)
{
    UVM_HINT_PARAMS params = {
        .base = (NvU64)begin,
        .length = end - begin,
        .destinationUuid = gpu_uuid_,
        .unloaded_chunks = 0,
        .rmStatus = 0,
    };
    int ret = ioctl(uvm_fd_, UVM_HINT, &params);
    if (ret == 0 && params.rmStatus == 0) return;
    XWARN("CUDA UVM Hint ioctl [0x%lx, 0x%lx] failed: ret = %d, status = 0x%x",
          begin, end, ret, params.rmStatus);
}

void CudaMemoryManager::PrefetchH2D(uint64_t begin, uint64_t end, bool sync)
{
    UVM_MIGRATE_PARAMS params = {
        .base = (NvU64)begin,
        .length = end - begin,
        .destinationUuid = gpu_uuid_,
        .flags = NvU32(sync ? 0 : 1),
        .semaphoreAddress = 0,
        .semaphorePayload = 0,
        .cpuNumaNode = 0,
        .userSpaceStart = 0,
        .userSpaceLength = 0,
        .rmStatus = 0,
    };
    int ret = ioctl(uvm_fd_, UVM_MIGRATE, &params);
    if (ret == 0 && params.rmStatus == 0) return;
    XWARN("CUDA UVM PrefetchH2D %s ioctl [0x%lx, 0x%lx] failed: ret = %d, status = 0x%x",
          sync ? "sync" : "async", begin, end, ret, params.rmStatus);
}
