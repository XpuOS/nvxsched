#pragma once

#include <functional>

#include "xsched/types.h"
#include "xsched/preempt/hal/hw_mem.h"
#include "xsched/preempt/hal/hw_queue.h"
#include "xsched/preempt/hal/hw_command.h"

namespace xsched::preempt
{

enum XQueueImplType
{
    kXQueueImplTypeAsync    = 0,
    kXQueueImplTypeBlocking = 1,
};

typedef int64_t XQueueFeatures;

class XQueue : public std::enable_shared_from_this<XQueue>
{
public:
    XQueue(XQueueImplType impl_type, XQueueFeatures features, std::shared_ptr<HwQueue> hwq)
        : kImplType(impl_type), kFeatures(features), kDevice(hwq->GetDevice())
        , kHandle((uint64_t)this ^ ((uint64_t)GetProcessId() << 48)), kHwQueue(hwq) {}
    virtual ~XQueue() = default;

    virtual void Submit(std::shared_ptr<HwCommand> hw_cmd) = 0;
    virtual std::shared_ptr<XQueueWaitAllCommand> SubmitWaitAll() = 0;
    virtual void WaitAll() = 0;
    virtual void Wait(std::shared_ptr<HwCommand> hw_cmd) = 0;
    virtual XQueueState Query() = 0;
    virtual int64_t GetHwCommandCount() = 0;

    /// @brief Suspend an XQueue and pause its execution.
    /// @param flags Suspending behavior flags, see XQueueSuspendFlag.
    /// @return true if successfully suspended, false if it is already suspended or terminated.
    virtual bool Suspend(int64_t flags) = 0;

    /// @brief Resume an XQueue and continue its execution.
    /// @param flags Resuming behavior flags, see XQueueResumeFlag.
    /// @return true if successfully resumed, false if it is already resumed.
    virtual bool Resume(int64_t flags) = 0;

    /// @brief Advise the memory context, i.e., memory regions accessed by the XQueue,
    /// within the given period of time.
    /// @param begin_ns The beginning time of the period in nanoseconds.
    /// Use 0 to indicate the current time.
    /// @param end_ns The ending time of the period in nanoseconds,
    /// Use UINT64_MAX to indicate the maximum known access time.
    virtual void AdviseMemory(XMemoryAdviseType advise, uint64_t begin_ns, uint64_t end_ns) = 0;

    /// @brief Update the memory information of the XQueue.
    /// A scheduling operation can suspend multiple XQueues,
    /// called after all given XQueues are suspended in a scheduling operation.
    /// @param operation_id The id of the scheduling operation.
    virtual void UpdateMemoryInfo(uint64_t operation_id) = 0;

    /// @brief Set the preempt level of the XQueue. NOT thread-safe, make sure there is
    /// NO other operations on the XQueue when calling this function.
    /// @param level The new preempt level.
    virtual void SetPreemptLevel(XPreemptLevel level) = 0;

    /// @brief Set the memory context type of the XQueue.
    /// @param ctx The new memory context type.
    virtual void SetMemoryContextType(XMemoryContextType ctx) = 0;

    /// @brief Set the launching configuration of the XQueue.
    /// @param threshold The new in-flight command threshold, <= 0 means no change.
    /// @param batch_size The new command batch size, <= 0 means no change.
    virtual void SetLaunchConfig(int64_t threshold, int64_t batch_size) = 0;

    XQueueImplType           GetImplType() const { return kImplType; }
    XDevice                  GetDevice()   const { return kDevice;   }
    XQueueHandle             GetHandle()   const { return kHandle;   }
    std::shared_ptr<HwQueue> GetHwQueue()  const { return kHwQueue;  }
    XQueueFeatures GetFeatures(XQueueFeatures features_mask = kQueueFeatureMaskAll) const
    { return kFeatures & features_mask; }

protected:
    const XQueueImplType           kImplType;
    const XQueueFeatures           kFeatures;
    const XDevice                  kDevice;
    const XQueueHandle             kHandle;
    const std::shared_ptr<HwQueue> kHwQueue;
};

class XQueueManager
{
public:
    STATIC_CLASS(XQueueManager);

    static XResult Add(XQueueHandle *xq_hp, HwQueueHandle hwq_h, int64_t level, int64_t flags);
    static XResult Del(XQueueHandle xq_h);
    static XResult Exists(XQueueHandle xq_h);
    static std::shared_ptr<XQueue> Get(XQueueHandle xq_h);

    static XResult ForEachWaitAll();
    static XResult ForEach(std::function<XResult(std::shared_ptr<XQueue>)> func);
    static XResult AutoCreate(std::function<XResult(HwQueueHandle *)> create_hwq);
    static XResult AutoDestroy(HwQueueHandle hwq_h);

private:
    static std::mutex mtx_;
    static std::unordered_map<XQueueHandle, std::shared_ptr<XQueue>> xqs_;
};

} // namespace xsched::preempt
