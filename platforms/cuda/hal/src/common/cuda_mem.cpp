#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <sys/ioctl.h>
#include <time.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdlib>

#include "cuxtra/cuxtra.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_mem.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cupti.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/predict/mem_access.h"
#include "xsched/utils/hash.h"
#include "xsched/utils/region.h"

#define CUDA_UVM_PAGE_SIZE 4096

using namespace xsched::cuda;
using namespace xsched::preempt;

namespace
{

constexpr uint64_t UVM_QUERY_CAPACITY = 8192; // hard code temporarily

uint64_t GetMonotonicRawNs()
{
    timespec ts = {};
    XASSERT(clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0,
            "clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

bool ContextUnavailable(CUresult result)
{
    return result == CUDA_ERROR_DEINITIALIZED || result == CUDA_ERROR_INVALID_CONTEXT ||
           result == CUDA_ERROR_CONTEXT_IS_DESTROYED;
}

// CUPTI shares name storage across activity records. Pointer keys are valid only
// inside one completion callback; never retain them across module/context lifetimes.
class KernelNameHashCache
{
public:
    KernelNameHashCache() { hashes_.reserve(64); }

    KernelNameHash Get(const char *name)
    {
        auto it = hashes_.find(name);
        if (it != hashes_.end()) return it->second;
        const auto hash = HashDjb2(name);
        if (hashes_.size() < 256) hashes_.emplace(name, hash);
        return hash;
    }

private:
    std::unordered_map<const char *, KernelNameHash> hashes_;
};

struct KernelExecutionRecord
{
    KernelNameHash hash;
    uint64_t start_ns, end_ns;

    auto Order() const { return std::tie(start_ns, end_ns, hash); }
    bool operator<(const KernelExecutionRecord &other) const { return Order() < other.Order(); }
    bool operator==(const KernelExecutionRecord &other) const { return Order() == other.Order(); }
};

void OrderKernelRecords(std::vector<KernelExecutionRecord> &records)
{
    std::sort(records.begin(), records.end());
    records.erase(std::unique(records.begin(), records.end()), records.end());
}

template<class FaultRange>
size_t AttributeKernelFaults(const std::vector<KernelExecutionRecord> &records,
                            std::vector<FaultRange> &faults, KernelMemoryMap &mapping)
{
    if (records.empty() || faults.empty()) return 0;
    std::sort(faults.begin(), faults.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.last_fault_service_timestamp_ns < rhs.last_fault_service_timestamp_ns;
    });
    size_t assignments = 0;
    auto first = faults.cbegin();
    for (const auto &record : records) {
        while (first != faults.cend() && first->last_fault_service_timestamp_ns < record.start_ns) ++first;
        // Map lookup is deferred until this kernel actually has a fault.
        // Scan independently so overlapping kernels retain all their matches.
        for (auto fault = first; fault != faults.cend(); ++fault) {
            if (fault->last_fault_service_timestamp_ns > record.end_ns) break;
            mapping[record.hash].second.Merge(fault->begin, fault->end);
            ++assignments;
        }
    }
    return assignments;
}

class KernelExecutionCollector
{
private:
    KernelExecutionCollector() {
        // HES avoids software timestamp instrumentation on supported Blackwell GPUs.
        // Keep the software path available for older CUPTI libraries and devices.
        const auto status = PTI::ActivityEnableHWTrace(1);
        const bool hardware_trace = status == CUPTI_SUCCESS;
        if (!hardware_trace) {
            XWARN("CUPTI hardware timestamp tracing unavailable (status=%d); using software tracing", status);
        }
        XINFO("fault collector timestamp mode: hardware=%d", hardware_trace);
        CUPTI_ASSERT(PTI::ActivityRegisterTimestampCallback(ReadTimestamp));
        CUPTI_ASSERT(PTI::ActivityRegisterCallbacks(PtiBufferRequested, PtiBufferCompleted));
        // Flush completed activities only at a suspended execution boundary.
    }

    ~KernelExecutionCollector() = default;

public:
    static void Start() {
        static const bool started = [] {
            Get();
            CUPTI_ASSERT(PTI::ActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
            started_.store(true, std::memory_order_release);
            XINFO("fault predictor kernel collection enabled");
            return true;
        }();
        UNUSED(started);
    }

    static bool IsStarted() { return started_.load(std::memory_order_acquire); }

    static bool Sync() {
        const CUresult sync_result = Driver::CtxSynchronize();
        if (ContextUnavailable(sync_result)) return false;
        CUDA_ASSERT(sync_result);
        CUPTI_ASSERT(PTI::ActivityFlushAll(0));
        return true;
    }

    static std::vector<KernelExecutionRecord> HarvestRecords() {
        auto &collector = Get();
        std::vector<KernelExecutionRecord> records;
        {
            std::lock_guard<std::mutex> lock(collector.records_mutex_);
            std::swap(records, collector.records_);
        }
        OrderKernelRecords(records);
        return records;
    }

private:
    static uint64_t CUPTIAPI ReadTimestamp()
    {
        return GetMonotonicRawNs();
    }

    static void CUPTIAPI PtiBufferRequested(uint8_t **buffer, size_t *size, size_t *max_num_records)
    {
        constexpr size_t kBufferSize = 8 * 1024 * 1024;
        *size = kBufferSize;
        *buffer = static_cast<uint8_t *>(std::malloc(kBufferSize));
        XASSERT(*buffer != nullptr, "failed to allocate CUPTI activity buffer");
        *max_num_records = 0; // no limit
    }

    static void CUPTIAPI PtiBufferCompleted(CUcontext ctx, uint32_t stream_id,
                                    uint8_t *buffer, size_t size, size_t valid_size)
    {
        UNUSED(size);
        size_t invalid_count = 0;
        std::vector<KernelExecutionRecord> batch;
        KernelNameHashCache name_hashes;
        batch.reserve(valid_size / sizeof(CUpti_ActivityKernel8));
        CUpti_Activity *record = nullptr;
        while (CUPTI_SUCCESS == PTI::ActivityGetNextRecord(buffer, valid_size, &record)) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                auto kernel_record = (const CUpti_ActivityKernel8 *)record;
                if (kernel_record->name == nullptr || kernel_record->start == 0 ||
                    kernel_record->end < kernel_record->start) {
                    ++invalid_count;
                    continue;
                }
                KernelExecutionRecord info = {
                    .hash = name_hashes.Get(kernel_record->name),
                    .start_ns = kernel_record->start,
                    .end_ns = kernel_record->end
                };
                batch.push_back(info);
            }
        }

        if (!batch.empty()) {
            auto &collector = Get();
            std::lock_guard<std::mutex> lock(collector.records_mutex_);
            if (collector.records_.empty()) collector.records_ = std::move(batch);
            else collector.records_.insert(collector.records_.end(), batch.begin(), batch.end());
        }
        if (invalid_count > 0) XWARN("CUPTI returned %zu kernel records with missing metadata/timestamps", invalid_count);

        size_t dropped = 0;
        CUPTI_ASSERT(PTI::ActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
        if (dropped > 0) XWARN("CUPTI dropped %zu activity records", dropped);
        std::free(buffer);
    }

private:
    static KernelExecutionCollector &Get()
    {
        // CUPTI may deliver callbacks during CUDA/global teardown.
        static auto *collector = new KernelExecutionCollector();
        return *collector;
    }

    std::mutex records_mutex_;
    std::vector<KernelExecutionRecord> records_;
    inline static std::atomic<bool> started_{false};

};

} // namespace

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

void CudaMemoryManager::UpdateMemoryInfo()
{
    if (!KernelExecutionCollector::IsStarted()) return;
    const CUresult push_result = Driver::CtxPushCurrent_v2(context_);
    if (ContextUnavailable(push_result)) return;
    CUDA_ASSERT(push_result);
    const bool synchronized = KernelExecutionCollector::Sync();
    CUcontext popped_context = nullptr;
    const CUresult pop_result = Driver::CtxPopCurrent_v2(&popped_context);
    if (ContextUnavailable(pop_result)) return;
    CUDA_ASSERT(pop_result);
    if (!synchronized) return;

    auto records = KernelExecutionCollector::HarvestRecords();
    if (records.empty()) return;
    std::vector<PageFaultRange> page_faults;
    const auto since_timestamp_ns = records.front().start_ns;
    if (!QueryPageFaultRecords(since_timestamp_ns, page_faults)) return;

    KernelMemoryMap kernel_mem_map;
    const auto assignments = AttributeKernelFaults(records, page_faults, kernel_mem_map);
    MemoryAccessPredictor::UpdateMemoryInfo(kernel_mem_map);
    XINFO("FaultPredictor update: kernels=%zu fault_ranges=%zu assignments=%zu",
          records.size(), page_faults.size(), assignments);
}

void CudaMemoryManager::OnContextTypeChange(XMemoryContextType type)
{
    if (type == kMemoryContextFault || type == kMemoryContextTemplateAndFault) {
        KernelExecutionCollector::Start();
    }
}

// **  Driver Wrapper  ** //

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

bool CudaMemoryManager::QueryPageFaultRecords(uint64_t since_timestamp_ns,
                                              std::vector<PageFaultRange> &records)
{
    std::vector<UVM_FAULT_INFO> fault_infos(UVM_QUERY_CAPACITY);
    UVM_QUERY_PF_PARAMS params = {
        .sinceTimestampNs = since_timestamp_ns,
        .rangeCapacity = fault_infos.size(),
        .rangeCount = 0,
        .faultRanges = fault_infos.data(),
        .rmStatus = 0,
    };

    int ret = ioctl(uvm_fd_, UVM_QUERY_PF, &params);
    uint64_t query_completed_timestamp_ns = GetMonotonicRawNs();
    if (ret != 0 || params.rmStatus != 0) {
        XWARN("CUDA UVM QueryPF ioctl failed: ret = %d, status = 0x%x", ret, params.rmStatus);
        return false;
    }
    if (params.rangeCount > fault_infos.size()) {
        XWARN("CUDA UVM QueryPF returned %zu records above capacity %zu",
              static_cast<size_t>(params.rangeCount), fault_infos.size());
        return false;
    }

    std::vector<PageFaultRange> result;
    result.reserve(params.rangeCount);
    for (size_t i = 0; i < params.rangeCount; ++i) {
        const UVM_FAULT_INFO &info = fault_infos[i];
        if (info.start >= info.end ||
            info.timestamp <= since_timestamp_ns ||
            info.timestamp > query_completed_timestamp_ns) {
            XWARN("CUDA UVM QueryPF returned invalid record %zu: "
                  "[0x%llx, 0x%llx), timestamp %llu, expected (%llu, %llu]",
                  i,
                  info.start,
                  info.end,
                  info.timestamp,
                  static_cast<unsigned long long>(since_timestamp_ns),
                  static_cast<unsigned long long>(query_completed_timestamp_ns));
            return false;
        }

        result.push_back({
            .begin = info.start,
            .end = info.end,
            .last_fault_service_timestamp_ns = info.timestamp,
        });
    }

    records = std::move(result);
    return true;
}
