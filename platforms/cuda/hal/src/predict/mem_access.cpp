#include <limits>
#include <fstream>
#include <algorithm>
#include <json/json.h>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/common.h"
#include "xsched/preempt/xmem/xmem.h"
#include "xsched/cuda/hal/common/cuda_mem.h"
#include "xsched/cuda/hal/predict/mem_access.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

MemoryAccessPredictor MemoryAccessPredictor::g_predictor;

MemoryAccessPredictor::MemoryAccessPredictor()
    : tmpl_pred_(GetMemoryContextTemplateFileName()), fault_pred_()
{
}

bool MemoryAccessPredictor::
PredictKernel(XMemoryContextType type, CUcontext ctx,
              std::shared_ptr<preempt::XMemoryManager> xmm,
              const KernelPredictionInfo &current,
              const KernelPredictionInfo &previous,
              utils::Region &reads, utils::Region &writes)
{
    if (xmm == nullptr) return false;

    switch (type) {
    case kMemoryContextUnknown:
    case kMemoryContextNone:
    case kMemoryContextAllocation:
        // Allocation context is handled by AsyncXQueue::AdviseMemory().
        return false;
    case kMemoryContextBuffer:
        return g_predictor.buf_pred_.PredictKernel(std::move(xmm), current, reads, writes);
    case kMemoryContextFault:
        return g_predictor.fault_pred_.PredictKernel(current, reads, writes);
    case kMemoryContextTemplate:
    case kMemoryContextTemplateAndBuffer:
    case kMemoryContextTemplateAndFault:
    {
        bool predicted = g_predictor.tmpl_pred_.PredictKernel(current, previous, reads, writes);
        if (DebugLogEnabled() && !predicted) {
            int32_t gdx, gdy, gdz;
            int32_t bdx, bdy, bdz;
            UnpackGridDim(current.gdim, gdx, gdy, gdz);
            UnpackBlockDim(current.bdim, bdx, bdy, bdz);
            const char *name = nullptr;
            cuXtraGetFunctionName(ctx, current.func, &name);
            XDEBG("failed to predict memory context using templates for kernel "
                  "gdim[%d, %d, %d] bdim[%d, %d, %d] %s",
                  gdx, gdy, gdz, bdx, bdy, bdz, name);
        }
        if (type == kMemoryContextTemplateAndBuffer && !predicted) {
            return g_predictor.buf_pred_.PredictKernel(std::move(xmm), current, reads, writes);
        }
        if (type == kMemoryContextTemplateAndFault) {
            bool fault_predicted = g_predictor.fault_pred_.PredictKernel(current, reads, writes);
            return predicted || fault_predicted;
        }
        return predicted;
    }
    default:
        XERRO("invalid memory context type: %d", type);
        return false;
    }
}

bool BufferBasedMemoryAccessPredictor::
PredictKernel(std::shared_ptr<preempt::XMemoryManager> xmm,
              const KernelPredictionInfo &current,
              utils::Region &reads, utils::Region &writes)
{
    for (size_t i = 0; i < current.param_sizes.size(); ++i) {
        size_t param_size = current.param_sizes[i];
        for (size_t offset = 0; offset + sizeof(void *) <= param_size; offset += sizeof(void *)) {
            // Scan all pointer parameters.
            void *ptr = *(void **)(uint64_t(current.params[i]) + offset);
            size_t size = xmm->GetAllocatedSize(ptr);
            if (size == 0) continue; // Not an allocated buffer.
            reads.Merge((uint64_t)ptr, (uint64_t)ptr + size);
            writes.Merge((uint64_t)ptr, (uint64_t)ptr + size);
        }
    }
    return true;
}

TemplateBasedMemoryAccessPredictor::TemplateBasedMemoryAccessPredictor(const std::string &filename)
{
    Json::Reader reader;
    std::ifstream file(filename);
    Json::Value mapping_json;
    reader.parse(file, mapping_json);
    file.close();

    for (auto it = mapping_json.begin(); it != mapping_json.end(); it++) {
        std::string key = it.name();
        KernelNameHash hash = HashDjb2(key.c_str());
        Json::Value &mappings = *it;
        XASSERT(mappings.isArray(), "mappings should be an array");
        for (auto mit = mappings.begin(); mit != mappings.end(); mit++) {
            Json::Value &mapping = *mit;
            auto &gdim = mapping["gdim"];
            auto &bdim = mapping["bdim"];
            XASSERT(gdim.isArray() && gdim.size() == 3, "gdim size should be 3");
            XASSERT(bdim.isArray() && bdim.size() == 3, "bdim size should be 3");
            GridDim gdim_pack = PackGridDim(gdim[0].asInt(), gdim[1].asInt(), gdim[2].asInt());
            BlockDim bdim_pack = PackBlockDim(bdim[0].asInt(), bdim[1].asInt(), bdim[2].asInt());

            XASSERT(mapping.isMember("read") && mapping["read"].isArray(),
                    "read should be an array");
            XASSERT(mapping.isMember("write") && mapping["write"].isArray(),
                    "write should be an array");
            Json::Value &reads = mapping["read"];
            Json::Value &writes = mapping["write"];
            for (auto read_it = reads.begin(); read_it != reads.end(); read_it++) {
                mappings_[hash][gdim_pack][bdim_pack].first.push_back(
                    AccessMapping::Create(*read_it));
            }
            for (auto write_it = writes.begin(); write_it != writes.end(); write_it++) {
                mappings_[hash][gdim_pack][bdim_pack].second.push_back(
                    AccessMapping::Create(*write_it));
            }
        }
    }
}

bool TemplateBasedMemoryAccessPredictor::
PredictKernel(const KernelPredictionInfo &current,
              const KernelPredictionInfo &previous,
              utils::Region &reads, utils::Region &writes)
{
    auto kit = mappings_.find(current.hash);
    if (kit == mappings_.end()) return false;
    auto git = kit->second.find(current.gdim);
    if (git == kit->second.end()) return false;
    auto bit = git->second.find(current.bdim);
    if (bit == git->second.end()) return false;
    auto &rw_mappings = bit->second;

    for (auto &mapping : rw_mappings.first) {
        mapping->MergeRegion(reads, previous.hash, previous.gdim, previous.bdim, previous.params, current.params);
    }
    for (auto &mapping : rw_mappings.second) {
        mapping->MergeRegion(writes, previous.hash, previous.gdim, previous.bdim, previous.params, current.params);
    }

    return true;
}

namespace
{
uint64_t FaultTokenParameter(const char *name, uint64_t fallback)
{
    const std::string value = GetEnv(name);
    if (value.empty()) return fallback;
    uint64_t parsed = 0;
    bool converted = false;
    const bool digits = std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (digits) {
        try { parsed = std::stoull(value); converted = true; } catch (...) {}
    }
    XASSERT(converted && parsed >= 1 && parsed <= 1000000,
            "%s must be an integer in [1, 1000000]", name);
    return parsed;
}

} // namespace

FaultDrivenMemoryAccessPredictor::FaultDrivenMemoryAccessPredictor()
    : initial_current_tokens_(FaultTokenParameter("XSCHED_FAULT_TOKEN_INITIAL_CURRENT", kInitialCurrentTokens))
    , initial_next_tokens_(FaultTokenParameter("XSCHED_FAULT_TOKEN_INITIAL_NEXT", kInitialNextTokens))
    , retry_multiplier_(FaultTokenParameter("XSCHED_FAULT_TOKEN_RETRY_MULTIPLIER", kRetryMultiplier))
{
    XINFO("fault predictor token parameters: initial_current=%lu initial_next=%lu retry_multiplier=%lu",
          initial_current_tokens_, initial_next_tokens_, retry_multiplier_);
}

bool FaultDrivenMemoryAccessPredictor::
PredictKernel(const KernelPredictionInfo &current,
              utils::Region &reads, utils::Region &writes)
{
    std::shared_ptr<const MemorySet> prediction;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = history_.find(current.hash);
        if (it == history_.end()) return false;

        auto &history = it->second;
        if (history.cached_generation != current_generation_ || history.cached_context == nullptr) {
            auto context = std::make_shared<MemorySet>();
            PredictPages(history.reads, current_generation_, context->reads);
            PredictPages(history.writes, current_generation_, context->writes);
            history.cached_context = std::move(context);
            history.cached_generation = current_generation_;
        }
        prediction = history.cached_context;
        const bool predicted = prediction->reads.RegionCount() != 0 || prediction->writes.RegionCount() != 0;
        if (history.reads.empty() && history.writes.empty()) history_.erase(it);
        if (!predicted) return false;
    }

    // Hold an immutable snapshot while merging, allowing updates to proceed concurrently.
    reads.Merge(prediction->reads);
    writes.Merge(prediction->writes);
    return true;
}

void FaultDrivenMemoryAccessPredictor::
UpdateMemoryInfo(const KernelMemoryMap &kernel_mem_map)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ++current_generation_;
    for (const auto &[hash, regions] : kernel_mem_map) {
        if (regions.first.RegionCount() == 0 && regions.second.RegionCount() == 0) continue;
        auto &history = history_[hash];
        UpdatePages(history.reads, regions.first, current_generation_);
        UpdatePages(history.writes, regions.second, current_generation_);
    }
}

void FaultDrivenMemoryAccessPredictor::
UpdatePages(PageTable &pages, const utils::Region &faults, uint64_t generation)
{
    faults.ForEach([&](uint64_t begin, uint64_t end) {
        if (begin >= end) return;
        // Include both partial blocks at the edges of a half-open fault range.
        const uint64_t last_page = (end - 1) / kPageSize;
        for (uint64_t page = begin / kPageSize; page <= last_page; ++page) {
            auto [it, inserted] = pages.try_emplace(page, PageInfo{generation, initial_current_tokens_, initial_next_tokens_});
            if (inserted) continue;

            PageInfo &info = it->second;
            const uint64_t age = generation - info.loaded_generation;
            if (age > info.current_tokens) {
                const uint64_t expired_age = age - info.current_tokens;
                if (expired_age >= info.next_tokens) {
                    info.current_tokens = initial_current_tokens_;
                    info.next_tokens = initial_next_tokens_;
                } else {
                    info.current_tokens = std::max(info.next_tokens - expired_age,
                                                   initial_next_tokens_);
                    constexpr uint64_t max_tokens = std::numeric_limits<uint64_t>::max();
                    info.next_tokens = info.current_tokens > max_tokens / retry_multiplier_
                        ? max_tokens : retry_multiplier_ * info.current_tokens;
                }
            }
            info.loaded_generation = generation;
        }
    });
}

bool FaultDrivenMemoryAccessPredictor::
PredictPages(PageTable &pages, uint64_t generation, utils::Region &region) const
{
    bool predicted = false;
    for (auto it = pages.begin(); it != pages.end();) {
        const auto &[page, info] = *it;
        // Compare ages rather than adding tokens to generation, avoiding overflow.
        const uint64_t age = generation - info.loaded_generation;
        if (age <= info.current_tokens) {
            const uint64_t begin = page * kPageSize;
            constexpr uint64_t max_address = std::numeric_limits<uint64_t>::max();
            const uint64_t end = begin > max_address - kPageSize
                ? max_address : begin + kPageSize;
            region.Merge(begin, end);
            predicted = true;
        } else if (age - info.current_tokens > info.next_tokens) {
            it = pages.erase(it);
            continue;
        }
        ++it;
    }
    return predicted;
}
