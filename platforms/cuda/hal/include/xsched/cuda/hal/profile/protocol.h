#pragma once

namespace xsched::cuda
{

enum MemAccessType
{
    kMemUnknown   = 0,
    kMemRead      = 0b01,
    kMemWrite     = 0b10,
};

} // namespace xsched::cuda
