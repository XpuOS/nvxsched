#pragma once

#include <list>
#include <chrono>
#include <memory>
#include <functional>

#include "xsched/types.h"
#include "xsched/sched/protocol/hint.h"
#include "xsched/sched/protocol/status.h"

namespace xsched::sched
{

typedef std::chrono::system_clock::time_point TimePoint;
typedef std::function<void (const TimePoint)> AddTimerFunc;
typedef std::function<void (const XQueueHandle)> OperateFunc;
typedef std::function<void (const XQueueHandle, const Timeslice)> SetTimesliceFunc;
typedef std::function<void (const std::list<XQueueHandle> &)> UpdateRunQueueFunc;

class Policy
{
public:
    Policy(XPolicyType type): kType(type) {}
    virtual ~Policy() = default;

    void SetSuspendFunc(OperateFunc suspend) { suspend_func_ = suspend; }
    void SetResumeFunc(OperateFunc resume) { resume_func_ = resume; }
    void SetAddTimerFunc(AddTimerFunc add_timer) { add_timer_func_ = add_timer; }
    void SetSetTimesliceFunc(SetTimesliceFunc set) { set_timeslice_func_ = set; }
    void SetUpdateRunQueueFunc(UpdateRunQueueFunc update) { update_run_queue_func_ = update; }

    virtual void Sched(const Status &status) = 0;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) = 0;

    const XPolicyType kType;

protected:
    void Suspend(XQueueHandle xqueue);
    void Resume(XQueueHandle xqueue);
    void AddTimer(const TimePoint time_point);
    void SetTimeslice(XQueueHandle handle, Timeslice timeslice);
    void UpdateRunQueue(const std::list<XQueueHandle> &run_queue);

private:
    OperateFunc suspend_func_ = nullptr;
    OperateFunc resume_func_ = nullptr;
    AddTimerFunc add_timer_func_ = nullptr;
    SetTimesliceFunc set_timeslice_func_ = nullptr;
    UpdateRunQueueFunc update_run_queue_func_ = nullptr;
};

std::unique_ptr<Policy> CreatePolicy(XPolicyType type);

} // namespace xsched::sched
