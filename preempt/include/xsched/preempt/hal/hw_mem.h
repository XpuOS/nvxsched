#pragma once

#include "xsched/types.h"
#include "xsched/utils/region.h"
#include "xsched/utils/common.h"

namespace xsched::preempt
{

class HwMemoryRequest
{
public:
    HwMemoryRequest(size_t size): kSize(size) {}
    virtual ~HwMemoryRequest() = default;

    size_t Size() const { return kSize; }

private:
    const size_t kSize = 0;
};

class HwMemoryManager
{
public:
    HwMemoryManager() = default;
    virtual ~HwMemoryManager() = default;

    virtual void *AllocatePagable(std::shared_ptr<HwMemoryRequest> req) = 0;
    virtual void Free(void *ptr) = 0;
    virtual void Advise(std::shared_ptr<utils::Region> mem, XMemoryAdviseType advise)
    {
        UNUSED(mem);
        UNUSED(advise);
    }
};

} // namespace xsched::preempt
