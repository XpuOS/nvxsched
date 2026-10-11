#pragma once

#include <mutex>
#include <unordered_map>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/hash.h"

namespace xsched::cuda
{

inline uint64_t GetKernelNameHash(CUcontext ctx, CUfunction func)
{
    static std::mutex mtx;
    static std::unordered_map<CUfunction, uint64_t> hash_map;

    if (ctx == nullptr || func == nullptr) return 0;
    uint64_t hash = 0;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = hash_map.find(func);
    if (it == hash_map.end()) {
        const char *name = nullptr;
        cuXtraGetFunctionName(ctx, func, &name);
        if (name == nullptr) return 0;
        hash = HashDjb2(name);
        hash_map[func] = hash;
    } else {
        hash = it->second;
    }

    return hash;
}

} // namespace xsched::cuda
