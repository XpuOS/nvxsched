#pragma once

#include "xsched/types.h"
#include "xsched/utils/region.h"

namespace xsched::preempt
{
struct MemorySet
{
    utils::Region reads;
    utils::Region writes;
};

} // namespace xsched::preempt
