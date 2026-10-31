#pragma once

#include <list>
#include <memory>
#include <cstdint>
#include <json/json.h>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/utils/region.h"

#ifndef UVM_PAGE_SIZE
    #define UVM_PAGE_SIZE   0x200000ULL // 2MB block size
    // #define UVM_PAGE_SIZE   0x1000ULL
#endif

namespace xsched::cuda
{

class IntReference
{
public:
    IntReference() = default;
    virtual ~IntReference() = default;
    virtual uint64_t GetValue(uint64_t name_hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
                              void **params_prev, void **params_cur) = 0;
    static std::unique_ptr<IntReference> Create(const Json::Value &ref);
};

class ImmediateInt : public IntReference
{
public:
    ImmediateInt(uint64_t value) : kValue(value) {}
    virtual ~ImmediateInt() = default;
    virtual uint64_t GetValue(uint64_t, uint64_t, uint32_t, void **, void **) override
    {
        return kValue;
    }

private:
    const uint64_t kValue;
};

inline uint64_t GetIntValue(void **params, size_t param_idx, size_t offset, size_t size)
{
    switch (size) {
        case 1:
            return *(uint8_t *)(((uint8_t *)(params[param_idx])) + offset);
        case 2:
            return *(uint16_t *)(((uint8_t *)(params[param_idx])) + offset);
        case 4:
            return *(uint32_t *)(((uint8_t *)(params[param_idx])) + offset);
        case 8:
            return *(uint64_t *)(((uint8_t *)(params[param_idx])) + offset);
        default:
            XERRO("invalid int size: %zu", size);
            return 0;
    }
}

class CurrentParameterInt : public IntReference
{
public:
    CurrentParameterInt(size_t param_idx, size_t offset, size_t size)
        : kParamIdx(param_idx), kOffset(offset), kSize(size) {}
    virtual ~CurrentParameterInt() = default;
    virtual uint64_t GetValue(uint64_t, uint64_t, uint32_t, void **, void **params_cur) override
    {
        return GetIntValue(params_cur, kParamIdx, kOffset, kSize);
    }

private:
    const size_t kParamIdx;
    const size_t kOffset;
    const size_t kSize;
};

class PreviousParameterInt : public IntReference
{
public:
    PreviousParameterInt(size_t param_idx, size_t offset, size_t size,
        uint64_t prev_kernel_name_hash, uint64_t prev_gdim, uint32_t prev_bdim)
        : kParamIdx(param_idx), kOffset(offset), kSize(size)
        , kPrevHash(prev_kernel_name_hash), kPrevGDim(prev_gdim), kPrevBBim(prev_bdim) {}
    virtual ~PreviousParameterInt() = default;
    virtual uint64_t GetValue(uint64_t name_hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
                              void **params_prev, void **) override
    {
        if (kPrevHash != name_hash_prev || pgdim_prev != kPrevGDim || pbdim_prev != kPrevBBim) {
            XWARN("previous kernel mismatch, "
                  "previous: 0x%lx, gdim 0x%lx, bdim 0x%x, expected: 0x%lx, gdim 0x%lx, bdim 0x%x",
                  name_hash_prev, pgdim_prev, pbdim_prev, kPrevHash, kPrevGDim, kPrevBBim);
            return 0;
        }
        return GetIntValue(params_prev, kParamIdx, kOffset, kSize);
    }

private:
    const size_t kParamIdx;
    const size_t kOffset;
    const size_t kSize;
    const uint64_t kPrevHash;
    const uint64_t kPrevGDim;
    const uint32_t kPrevBBim;
};

// Mapping between memory access and kernel parameters

enum MappingType
{
    kMappingUnknown                                         = 0,
    kMappingExactBeginFixedSize                             = 1,
    kMappingExactBeginLinearSize                            = 2,
    kMappingExactBeginMultipliedSize                        = 3,
    kMappingExactEndFixedSize                               = 4,
    kMappingExactBeginFixedSizeFixedStrideFixedCount        = 5,
    kMappingExactBeginFixedSizeExact3DStrideExact3DCount    = 6,
    kMappingExactBeginLinearSizeFixedStrideFixedCount       = 7,
    kMappingExactBeginLinearSizeLinear3DStrideLinear3DCount = 8,
};

class AccessMapping
{
public:
    AccessMapping(MappingType type) : kType(type) {}
    virtual ~AccessMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t kernel_name_hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) = 0;
    
    static std::unique_ptr<AccessMapping> Create(Json::Value &mapping);

protected:
    const MappingType kType;
};

class ExactBeginFixedSizeMapping : public AccessMapping
{
public:
    ExactBeginFixedSizeMapping(Json::Value &mapping);
    virtual ~ExactBeginFixedSizeMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override
    {
        uint64_t begin, end;
        begin = begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        end = begin + size_;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    size_t size_ = 0;
};

class ExactBeginLinearSizeMapping : public AccessMapping
{
public:
    ExactBeginLinearSizeMapping(Json::Value &mapping);
    virtual ~ExactBeginLinearSizeMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override
    {
        uint64_t begin, end;
        begin = begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        size_t size = size_multiplier_ * size_ref_->GetValue(
            hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        end = begin + size;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    std::unique_ptr<IntReference> size_ref_ = nullptr;
    double size_multiplier_ = 0;
};

class ExactBeginMultipliedSizeMapping : public AccessMapping
{
public:
    ExactBeginMultipliedSizeMapping(Json::Value &mapping);
    virtual ~ExactBeginMultipliedSizeMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override
    {
        uint64_t begin, end;
        begin = begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        double size = size_multiplier_;
        for (auto &size_ref : size_refs_) {
            size *= size_ref->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        }
        end = begin + (size_t)size;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    std::list<std::unique_ptr<IntReference>> size_refs_;
    double size_multiplier_ = 0;
};

class ExactEndFixedSizeMapping : public AccessMapping
{
public:
    ExactEndFixedSizeMapping(Json::Value &mapping);
    virtual ~ExactEndFixedSizeMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override
    {
        uint64_t begin, end;
        end = end_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        begin = end - size_;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }

private:
    std::unique_ptr<IntReference> end_ref_ = nullptr;
    size_t size_ = 0;
};

class ExactBeginFixedSizeFixedStrideFixedCountMapping : public AccessMapping
{
public:
    ExactBeginFixedSizeFixedStrideFixedCountMapping(Json::Value &mapping);
    virtual ~ExactBeginFixedSizeFixedStrideFixedCountMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    size_t size_ = 0;
    size_t stride_ = 0;
    size_t count_ = 0;
};

class ExactBeginFixedSizeExact3DStrideExact3DCountMapping : public AccessMapping
{
public:
    ExactBeginFixedSizeExact3DStrideExact3DCountMapping(Json::Value &mapping);
    virtual ~ExactBeginFixedSizeExact3DStrideExact3DCountMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    size_t size_ = 0;
    std::unique_ptr<IntReference> strides_[3] = {nullptr, nullptr, nullptr};
    std::unique_ptr<IntReference> counts_[3] = {nullptr, nullptr, nullptr};
};

class ExactBeginLinearSizeFixedStrideFixedCountMapping : public AccessMapping
{
public:
    ExactBeginLinearSizeFixedStrideFixedCountMapping(Json::Value &mapping);
    virtual ~ExactBeginLinearSizeFixedStrideFixedCountMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    std::unique_ptr<IntReference> size_ref_ = nullptr;
    double size_multiplier_ = 0;
    size_t stride_ = 0;
    size_t count_ = 0;
};

class ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping : public AccessMapping
{
public:
    ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping(Json::Value &mapping);
    virtual ~ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    std::unique_ptr<IntReference> size_ref_ = nullptr;
    std::unique_ptr<IntReference> strides_[3] = {nullptr, nullptr, nullptr};
    std::unique_ptr<IntReference> counts_[3] = {nullptr, nullptr, nullptr};
    double size_multiplier_ = 0;
    double strides_multipliers_[3] = {0, 0, 0};
    double counts_multipliers_[3] = {0, 0, 0};
};

} // namespace xsched::cuda
