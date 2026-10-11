#include "xsched/preempt/xmem/xmem.h"

using namespace xsched::utils;
using namespace xsched::preempt;

void *XMemoryManager::Allocate(std::shared_ptr<HwMemoryRequest> req)
{
    std::lock_guard<std::mutex> lock(mtx_);
    void *ptr = hw_mm_->AllocatePagable(req);
    if (ptr == nullptr) return ptr;
    size_t size = req->Size();
    allocated_[ptr] = size;
    return ptr;
}

void XMemoryManager::Free(void *ptr)
{
    std::lock_guard<std::mutex> lock(mtx_);
    hw_mm_->Free(ptr);
    allocated_.erase(ptr);
}

void XMemoryManager::Advise(std::shared_ptr<utils::Region> mem, XMemoryAdviseType advise)
{
    XDEBG("XMemoryManager::Advise %zu MB type %d", mem->Size() / 1024 / 1024, advise);
    hw_mm_->Advise(mem, advise);
}

size_t XMemoryManager::GetTotalAllocatedSize()
{
    size_t total_size = 0;
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &[ptr, size] : allocated_) {
        total_size += size;
    }
    return total_size;
}

size_t XMemoryManager::GetAllocatedSize(void *ptr)
{
    std::lock_guard<std::mutex> lock(mtx_);

    // Use upper_bound to find the first element strictly greater than ptr.
    auto it = allocated_.upper_bound(ptr);
    if (it == allocated_.begin()) return 0;

    // Decrement to get the allocation with the largest address <= ptr.
    --it;

    uintptr_t target = (uintptr_t)ptr;
    uintptr_t base = (uintptr_t)it->first;
    size_t size = it->second;

    if (target < base || target >= base + size) return 0;
    return (base + size) - target;
}

std::shared_ptr<Region> XMemoryManager::GetAllocatedMemory()
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto region = std::make_shared<Region>();
    for (auto &[ptr, size] : allocated_) {
        region->Merge((uint64_t)ptr, (uint64_t)ptr + size);
    }
    return region;
}
