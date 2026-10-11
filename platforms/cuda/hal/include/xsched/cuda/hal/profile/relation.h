#pragma once

#include <list>
#include <memory>
#include <cstdint>
#include <json/json.h>

#include "xsched/utils/log.h"
#include "xsched/utils/common.h"
#include "xsched/utils/region.h"

#define UVM_PAGE_SIZE   0x200000ULL
// #define UVM_PAGE_SIZE   0x1000ULL

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

// Relation between memory access and kernel parameters

enum RelationType
{
    kRelationUnknown                                         = 0,
    kRelationExactBeginFixedSize                             = 1,
    kRelationExactBeginLinearSize                            = 2,
    kRelationExactBeginMultipliedSize                        = 3,
    kRelationExactEndFixedSize                               = 4,
    kRelationExactBeginFixedSizeFixedStrideFixedCount        = 5,
    kRelationExactBeginFixedSizeExact3DStrideExact3DCount    = 6,
    kRelationExactBeginLinearSizeFixedStrideFixedCount       = 7,
    kRelationExactBeginLinearSizeLinear3DStrideLinear3DCount = 8,
};

class AccessRelation
{
public:
    AccessRelation(RelationType type) : kType(type) {}
    virtual ~AccessRelation() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t kernel_name_hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) = 0;
    
    static std::unique_ptr<AccessRelation> Create(Json::Value &relation);

protected:
    const RelationType kType;
};

class ExactBeginFixedSizeRelation : public AccessRelation
{
public:
    ExactBeginFixedSizeRelation(Json::Value &relation);
    virtual ~ExactBeginFixedSizeRelation() = default;
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

class ExactBeginLinearSizeRelation : public AccessRelation
{
public:
    ExactBeginLinearSizeRelation(Json::Value &relation);
    virtual ~ExactBeginLinearSizeRelation() = default;
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

class ExactBeginMultipliedSizeRelation : public AccessRelation
{
public:
    ExactBeginMultipliedSizeRelation(Json::Value &relation);
    virtual ~ExactBeginMultipliedSizeRelation() = default;
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

class ExactEndFixedSizeRelation : public AccessRelation
{
public:
    ExactEndFixedSizeRelation(Json::Value &relation);
    virtual ~ExactEndFixedSizeRelation() = default;
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

class ExactBeginFixedSizeFixedStrideFixedCountRelation : public AccessRelation
{
public:
    ExactBeginFixedSizeFixedStrideFixedCountRelation(Json::Value &relation);
    virtual ~ExactBeginFixedSizeFixedStrideFixedCountRelation() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    size_t size_ = 0;
    size_t stride_ = 0;
    size_t count_ = 0;
};

class ExactBeginFixedSizeExact3DStrideExact3DCountRelation : public AccessRelation
{
public:
    ExactBeginFixedSizeExact3DStrideExact3DCountRelation(Json::Value &relation);
    virtual ~ExactBeginFixedSizeExact3DStrideExact3DCountRelation() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    size_t size_ = 0;
    std::unique_ptr<IntReference> strides_[3] = {nullptr, nullptr, nullptr};
    std::unique_ptr<IntReference> counts_[3] = {nullptr, nullptr, nullptr};
};

class ExactBeginLinearSizeFixedStrideFixedCountRelation : public AccessRelation
{
public:
    ExactBeginLinearSizeFixedStrideFixedCountRelation(Json::Value &relation);
    virtual ~ExactBeginLinearSizeFixedStrideFixedCountRelation() = default;
    virtual void MergeRegion(utils::Region &region, uint64_t hash_prev,
        uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev, void **params_cur) override;

private:
    std::unique_ptr<IntReference> begin_ref_ = nullptr;
    std::unique_ptr<IntReference> size_ref_ = nullptr;
    double size_multiplier_ = 0;
    size_t stride_ = 0;
    size_t count_ = 0;
};

class ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation : public AccessRelation
{
public:
    ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation(Json::Value &relation);
    virtual ~ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation() = default;
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
