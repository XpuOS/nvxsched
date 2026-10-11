#pragma once

#include <map>
#include <mutex>
#include <memory>

#include "xsched/types.h"
#include "xsched/utils/env.h"
#include "xsched/protocol/names.h"
#include "xsched/preempt/xmem/rw.h"
#include "xsched/preempt/hal/hw_mem.h"

namespace xsched::preempt
{

inline std::string GetExecutionTimeFileName()
{
    static const std::string file = []{
        std::string str = GetEnv(XSCHED_EXECUTION_TIME_FILE_ENV_NAME);
        return str.empty() ? std::string("execution_time.json") : str;
    }();
    return file;
}

inline std::string GetMemoryContextHeuristicFileName()
{
    static const std::string file = []{
        std::string str = GetEnv(XSCHED_MEMORY_CONTEXT_HEURISTIC_FILE_ENV_NAME);
        return str.empty() ? std::string("memory_context.json") : str;
    }();
    return file;
}

class XMemoryManager
{
public:
    XMemoryManager(std::shared_ptr<HwMemoryManager> hw_mm): hw_mm_(hw_mm) {}
    virtual ~XMemoryManager() = default;
    std::shared_ptr<HwMemoryManager> GetHwMemoryManager() { return hw_mm_; }

    void *Allocate(std::shared_ptr<HwMemoryRequest> req);
    void Free(void *ptr);
    void Advise(std::shared_ptr<utils::Region> mem, XMemoryAdviseType advise);

    size_t GetTotalAllocatedSize();
    size_t GetAllocatedSize(void *ptr);
    std::shared_ptr<utils::Region> GetAllocatedMemory();

private:
    std::mutex mtx_;
    std::map<void *, size_t> allocated_;
    std::shared_ptr<HwMemoryManager> hw_mm_ = nullptr;
};

} // namespace xsched::preempt
