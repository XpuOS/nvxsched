#pragma once

#include <list>
#include <chrono>

#include "xsched/types.h"
#include "xsched/protocol/def.h"
#include "xsched/sched/policy/policy.h"
#include "xsched/sched/protocol/hint.h"

namespace xsched::sched
{

class StrictProcessUtilizationPartitionPolicy : public Policy
{
public:
    StrictProcessUtilizationPartitionPolicy();
    virtual ~StrictProcessUtilizationPartitionPolicy() = default;

    virtual void Sched(const Status &status) override;
    virtual void RecvHint(std::shared_ptr<const Hint> hint) override;

private:
    void RoundRobin(const Status &status);
    uint64_t GetOverheadUs(const Status &status, PID pid);

    std::set<PID> pid_in_queue_;
    std::list<PID> run_queue_;
    std::map<PID, Utilization> utils_;
    std::chrono::system_clock::time_point timeslice_end_;
    // for overhead compensation
    bool compensation_started_ = false;
    std::chrono::system_clock::time_point compensation_end_;
    Timeslice timeslice_avg_us_ = TIMESLICE_DEFAULT;
};

} // namespace xsched::sched
