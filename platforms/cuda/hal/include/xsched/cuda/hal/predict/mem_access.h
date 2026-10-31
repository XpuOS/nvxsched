#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>

#include "xsched/types.h"
#include "xsched/utils/region.h"
#include "xsched/preempt/xmem/xmem.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/predict/mapping.h"

namespace xsched::cuda
{

class BufferBasedMemoryAccessPredictor
{
public:
    BufferBasedMemoryAccessPredictor() = default;
    ~BufferBasedMemoryAccessPredictor() = default;

    bool PredictKernel(std::shared_ptr<preempt::XMemoryManager> xmm,
                       const KernelPredictionInfo &current,
                       utils::Region &reads, utils::Region &writes);
};

class TemplateBasedMemoryAccessPredictor
{
public:
    TemplateBasedMemoryAccessPredictor(const std::string &filename);
    ~TemplateBasedMemoryAccessPredictor() = default;

    bool PredictKernel(const KernelPredictionInfo &current,
                       const KernelPredictionInfo &previous,
                       utils::Region &reads, utils::Region &writes);

private:
    // mappings_[kernel][gdim][bdim] = {read_mapping[], write_mapping[]}, read-only
    std::unordered_map<KernelNameHash, std::map<GridDim, std::map<BlockDim, std::pair<
        std::vector<std::unique_ptr<AccessMapping>>,
        std::vector<std::unique_ptr<AccessMapping>>>>>> mappings_;
};

// Complete CUDA symbol hash -> {reads, writes}; dimensions and parameters are excluded.
using KernelMemoryMap = std::unordered_map<KernelNameHash, std::pair<utils::Region, utils::Region>>;

class FaultDrivenMemoryAccessPredictor
{
public:
    FaultDrivenMemoryAccessPredictor();
    ~FaultDrivenMemoryAccessPredictor() = default;

    bool PredictKernel(const KernelPredictionInfo &current,
                       utils::Region &reads, utils::Region &writes);
    // Each call advances one generation, including an empty observation map.
    void UpdateMemoryInfo(const KernelMemoryMap &kernel_mem_map);

private:
    static constexpr uint64_t kPageSize = 1ULL << 21;
    static constexpr uint64_t kInitialCurrentTokens = 4;
    static constexpr uint64_t kInitialNextTokens = 8;
    static constexpr uint64_t kRetryMultiplier = 8;
    struct PageInfo
    {
        uint64_t loaded_generation;
        uint64_t current_tokens;
        uint64_t next_tokens;
    };
    using PageTable = std::unordered_map<uint64_t, PageInfo>;

    struct KernelHistory
    {
        PageTable reads;
        PageTable writes;
        // Built lazily once per generation; an empty result is cached as well.
        uint64_t cached_generation = 0;
        std::shared_ptr<const preempt::MemorySet> cached_context;
    };

    void UpdatePages(PageTable &pages, const utils::Region &faults, uint64_t generation);
    bool PredictPages(PageTable &pages, uint64_t generation, utils::Region &region) const;

    std::mutex mutex_;
    uint64_t current_generation_ = 0;
    const uint64_t initial_current_tokens_;
    const uint64_t initial_next_tokens_;
    const uint64_t retry_multiplier_;
    std::unordered_map<KernelNameHash, KernelHistory> history_;
};

class MemoryAccessPredictor
{
public:
    MemoryAccessPredictor();
    ~MemoryAccessPredictor() = default;
    static bool PredictKernel(XMemoryContextType type, CUcontext ctx,
                              std::shared_ptr<preempt::XMemoryManager> xmm,
                              const KernelPredictionInfo &current,
                              const KernelPredictionInfo &previous,
                              utils::Region &reads, utils::Region &writes);
    static void UpdateMemoryInfo(const KernelMemoryMap &kernel_mem_map)
    {
        return g_predictor.fault_pred_.UpdateMemoryInfo(kernel_mem_map);
    }

private:
    static MemoryAccessPredictor g_predictor;

    BufferBasedMemoryAccessPredictor buf_pred_;
    TemplateBasedMemoryAccessPredictor tmpl_pred_;
    FaultDrivenMemoryAccessPredictor fault_pred_;
};

} // namespace xsched::cuda
