#include <cstdlib>

#include "xsched/types.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/policy/hpf.h"
#include "xsched/sched/policy/hhpf.h"
#include "xsched/sched/policy/chpf.h"
#include "xsched/sched/policy/up.h"
#include "xsched/sched/policy/pup.h"
#include "xsched/sched/policy/spup.h"
#include "xsched/sched/policy/kedf.h"
#include "xsched/sched/policy/lax.h"
#include "xsched/sched/policy/awf.h"
#include "xsched/sched/policy/cfs.h"
#include "xsched/sched/policy/mlfq.h"
// NEW_POLICY: New policy headers go here.

using namespace xsched::sched;

void Policy::Suspend(XQueueHandle xqueue)
{
    if (suspend_func_) return suspend_func_(xqueue);
    XDEBG("suspend function not set");
}

void Policy::Resume(XQueueHandle xqueue)
{
    if (resume_func_) return resume_func_(xqueue);
    XDEBG("resume function not set");
}

void Policy::AddTimer(const TimePoint time_point)
{
    if (add_timer_func_) return add_timer_func_(time_point);
    XDEBG("add timer function not set");
}

void Policy::SetTimeslice(XQueueHandle handle, Timeslice timeslice)
{
    if (set_timeslice_func_) return set_timeslice_func_(handle, timeslice);
    XDEBG("set timeslice function not set");
}

void Policy::UpdateRunQueue(const std::list<XQueueHandle> &run_queue)
{
    if (update_run_queue_func_) return update_run_queue_func_(run_queue);
    XDEBG("update run queue function not set");
}

std::unique_ptr<Policy> xsched::sched::CreatePolicy(XPolicyType type)
{
    // NEW_POLICY: A new case handling new PolicyType should be added here
    // when creating a new policy.
    switch (type) {
        case kPolicyHighestPriorityFirst:
            return std::make_unique<HighestPriorityFirstPolicy>();
        case kPolicyHeterogeneousHighestPriorityFirst:
            return std::make_unique<HeterogeneousHighestPriorityFirstPolicy>();
        case kPolicyCPUHighestPriorityFirst:
            return std::make_unique<CPUHighestPriorityFirstPolicy>();
        case kPolicyUtilizationPartition:
            return std::make_unique<UtilizationPartitionPolicy>();
        case kPolicyProcessUtilizationPartition:
            return std::make_unique<ProcessUtilizationPartitionPolicy>();
        case kPolicyStrictProcessUtilizationPartition:
            return std::make_unique<StrictProcessUtilizationPartitionPolicy>();
        case kPolicyKEarliestDeadlineFirst:
            return std::make_unique<KEarliestDeadlineFirstPolicy>();
        case kPolicyLaxity:
            return std::make_unique<LaxityPolicy>();
        case kPolicyActiveWindowFirst:
            return std::make_unique<ActiveWindowFirstPolicy>();
        case kPolicyCompletelyFairScheduler:
            return std::make_unique<CompletelyFairSchedulerPolicy>();
        case kPolicyMultiLevelFeedbackQueue:
            return std::make_unique<MultiLevelFeedbackQueuePolicy>();
        // NEW_POLICY: New PolicyTypes handling goes here.
        default:
            XASSERT(false, "invalid policy type: %d", type);
            return nullptr;
    }
}
