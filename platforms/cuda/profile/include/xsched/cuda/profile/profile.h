#pragma once

#include <string>

#include "xsched/protocol/def.h"
#include "xsched/cuda/hal/common/cupti.h"
#include "xsched/cuda/hal/common/driver.h"

namespace xsched::cuda
{

inline size_t GetWarmupCnt()
{
    char *env = std::getenv(XSCHED_EXECUTION_TIME_WARMUP_COUNT_ENV_NAME);
    if (env == nullptr || strlen(env) == 0) return 1000;
    return std::stoul(env);
}

inline size_t GetProfileCnt()
{
    char *env = std::getenv(XSCHED_EXECUTION_TIME_PROFILE_COUNT_ENV_NAME);
    if (env == nullptr || strlen(env) == 0) return 5000;
    size_t cnt = std::stoul(env);
    XASSERT(cnt > 0, "env %s must be positive", XSCHED_EXECUTION_TIME_PROFILE_COUNT_ENV_NAME);
    return std::stoul(env);
}

CUresult XInit(unsigned int Flags);

void PtiInit();
void PtiFini();
void PtiSave();
void PtiRecvKernelRecord(CUpti_ActivityKernel6 *kernel);
void PtiRecvMemcpyRecord(CUpti_ActivityMemcpy6 *memcpy);
void PtiRecvMemsetRecord(CUpti_ActivityMemset4 *memset);

} // namespace xsched::cuda
