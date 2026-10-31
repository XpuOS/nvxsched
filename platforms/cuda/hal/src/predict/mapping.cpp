#include <cstddef>
#include <cstdint>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/hash.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/predict/mapping.h"

using namespace xsched::cuda;
using namespace xsched::utils;

std::unique_ptr<IntReference> IntReference::Create(const Json::Value &ref)
{
    if (ref.isIntegral()) return std::make_unique<ImmediateInt>(ref.asUInt64());
    XASSERT(ref.isMember("type") && ref["type"].isString(), "type should be a string");

    if (ref["type"] == "CURRENT_KERNEL") {
        return std::make_unique<CurrentParameterInt>(
            ref["param"].asUInt64(), ref["offset"].asUInt64(), ref["size"].asUInt64());
    } else { // ref["type"] == "PREVIOUS_KERNEL"
        auto &gdim = ref["gdim"];
        auto &bdim = ref["bdim"];
        XASSERT(gdim.isArray() && gdim.size() == 3, "gdim size should be 3");
        XASSERT(bdim.isArray() && bdim.size() == 3, "bdim size should be 3");
        uint64_t gdim_pack = PackGridDim(gdim[0].asInt(), gdim[1].asInt(), gdim[2].asInt());
        uint32_t bdim_pack = PackBlockDim(bdim[0].asInt(), bdim[1].asInt(), bdim[2].asInt());
        return std::make_unique<PreviousParameterInt>(
            ref["param"].asUInt64(), ref["offset"].asUInt64(), ref["size"].asUInt64(),
            HashDjb2(ref["kernel"].asCString()), gdim_pack, bdim_pack);
    }
}

std::unique_ptr<AccessMapping> AccessMapping::Create(Json::Value &mapping)
{
    XASSERT(mapping.isMember("type") && mapping["type"].isString(), "type should be a string");
    if (mapping["type"] == "EXACT_BEGIN_FIXED_SIZE") {
        return std::make_unique<ExactBeginFixedSizeMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_LINEAR_SIZE") {
        return std::make_unique<ExactBeginLinearSizeMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_MULTIPLIED_SIZE") {
        return std::make_unique<ExactBeginMultipliedSizeMapping>(mapping);
    } else if (mapping["type"] == "EXACT_END_FIXED_SIZE") {
        return std::make_unique<ExactEndFixedSizeMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_FIXED_SIZE_FIXED_STRIDE_FIXED_COUNT") {
        return std::make_unique<ExactBeginFixedSizeFixedStrideFixedCountMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_FIXED_SIZE_EXACT_3D_STRIDE_EXACT_3D_COUNT") {
        return std::make_unique<ExactBeginFixedSizeExact3DStrideExact3DCountMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_LINEAR_SIZE_FIXED_STRIDE_FIXED_COUNT") {
        return std::make_unique<ExactBeginLinearSizeFixedStrideFixedCountMapping>(mapping);
    } else if (mapping["type"] == "EXACT_BEGIN_LINEAR_SIZE_LINEAR_3D_STRIDE_LINEAR_3D_COUNT") {
        return std::make_unique<ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping>(mapping);
    }

    XERRO("unknown mapping type: %s", mapping["type"].asString().c_str());
    return nullptr;
}

ExactBeginFixedSizeMapping::ExactBeginFixedSizeMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginFixedSize)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ = mapping["size"].asUInt64();
}

ExactBeginLinearSizeMapping::ExactBeginLinearSizeMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginLinearSize)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ref_ = IntReference::Create(mapping["size"]);
    size_multiplier_ = mapping["size"]["multiplier"].asDouble();
}

ExactBeginMultipliedSizeMapping::ExactBeginMultipliedSizeMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginMultipliedSize)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_multiplier_ = mapping["size"]["multiplier"].asDouble();
    for (auto &multiplier : mapping["size"]["multipliers"]) {
        size_refs_.push_back(IntReference::Create(multiplier));
    }
}

ExactEndFixedSizeMapping::ExactEndFixedSizeMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactEndFixedSize)
{
    mapping["end"]["size"] = sizeof(void *);
    end_ref_ = IntReference::Create(mapping["end"]);
    size_ = mapping["size"].asUInt64();
}

ExactBeginFixedSizeFixedStrideFixedCountMapping::
ExactBeginFixedSizeFixedStrideFixedCountMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginFixedSizeFixedStrideFixedCount)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ = mapping["size"].asUInt64();
    stride_ = mapping["stride"].asUInt64();
    count_ = mapping["count"].asUInt64();
}

void ExactBeginFixedSizeFixedStrideFixedCountMapping::
MergeRegion(utils::Region &region, uint64_t hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
            void **params_prev, void **params_cur)
{
    for (size_t i = 0; i < count_; i++) {
        uint64_t begin = i * stride_;
        begin += begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        uint64_t end = begin + size_;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }
}

ExactBeginFixedSizeExact3DStrideExact3DCountMapping::
ExactBeginFixedSizeExact3DStrideExact3DCountMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginFixedSizeExact3DStrideExact3DCount)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ = mapping["size"].asUInt64();
    for (int i = 0; i < 3; i++) {
        strides_[i] = IntReference::Create(mapping["stride"][i]);
        counts_[i] = IntReference::Create(mapping["count"][i]);
    }
}

void ExactBeginFixedSizeExact3DStrideExact3DCountMapping::
MergeRegion(utils::Region &region, uint64_t hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
            void **params_prev, void **params_cur)
{
    size_t strides[3];
    size_t counts[3];
    uint64_t begin;
    begin = begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);

    for (int i = 0; i < 3; i++) {
        strides[i] = strides_[i]->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        counts[i] = counts_[i]->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
    }
    
    for (int i = 2; i >= 0; i--) {
        if (strides[i] * counts[i] <= UVM_PAGE_SIZE) {
            size_ = (counts[i] - 1) * strides[i] + size_;
            strides[i] = 0;
            counts[i] = 1;
        }
    }

    for (size_t i = 0; i < counts[2]; i++) {
        for (size_t j = 0; j < counts[1]; j++) {
            for (size_t k = 0; k < counts[0]; k++) {
                uint64_t region_begin = begin + i * strides[2] + j * strides[1] + k * strides[0];
                uint64_t region_end = region_begin + size_;
                region.Merge(ROUND_DOWN(region_begin, UVM_PAGE_SIZE),
                         ROUND_UP(region_end, UVM_PAGE_SIZE));
                // region.Merge(region_begin, region_end);
            }
        }
    }
}

ExactBeginLinearSizeFixedStrideFixedCountMapping::
ExactBeginLinearSizeFixedStrideFixedCountMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginLinearSizeFixedStrideFixedCount)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ref_ = IntReference::Create(mapping["size"]);
    size_multiplier_ = mapping["size"]["multiplier"].asDouble();
    stride_ = mapping["stride"].asUInt64();
    count_ = mapping["count"].asUInt64();
}

void ExactBeginLinearSizeFixedStrideFixedCountMapping::
MergeRegion(utils::Region &region, uint64_t hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
            void **params_prev, void **params_cur)
{
    size_t size = size_multiplier_ * size_ref_->GetValue(
        hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
    for (size_t i = 0; i < count_; i++) {
        uint64_t begin = i * stride_;
        begin += begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        uint64_t end = begin + size;
        // region.Merge(begin, end);
        region.Merge(ROUND_DOWN(begin, UVM_PAGE_SIZE), ROUND_UP(end, UVM_PAGE_SIZE));
    }
}

ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping::
ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping(Json::Value &mapping)
    : AccessMapping(kMappingExactBeginLinearSizeLinear3DStrideLinear3DCount)
{
    mapping["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(mapping["begin"]);
    size_ref_ = IntReference::Create(mapping["size"]);
    size_multiplier_ = mapping["size"]["multiplier"].asDouble();
    for (int i = 0; i < 3; i++) {
        strides_[i] = IntReference::Create(mapping["stride"][i]);
        counts_[i] = IntReference::Create(mapping["count"][i]);
        strides_multipliers_[i] = mapping["stride"][i]["multiplier"].asDouble();
        counts_multipliers_[i] = mapping["count"][i]["multiplier"].asDouble();
    }
}

void ExactBeginLinearSizeLinear3DStrideLinear3DCountMapping::
MergeRegion(utils::Region &region, uint64_t hash_prev, uint64_t pgdim_prev, uint32_t pbdim_prev,
            void **params_prev, void **params_cur)
{
    size_t strides[3];
    size_t counts[3];
    uint64_t begin, size;
    begin = begin_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
    size = size_multiplier_ * size_ref_->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);

    for (int i = 0; i < 3; i++) {
        strides[i] = strides_[i]->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        counts[i] = counts_[i]->GetValue(hash_prev, pgdim_prev, pbdim_prev, params_prev, params_cur);
        strides[i] *= strides_multipliers_[i];
        counts[i] *= counts_multipliers_[i];
    }

    // Optimization: if the stride is less than the page size, then the adjacent two regions
    // must be in the same or adjacent pages, thus we can merge all regions into a single one.
    for (int i = 2; i >= 0; i--) {
        if (strides[i] <= UVM_PAGE_SIZE) {
            size = (counts[i] - 1) * strides[i] + size;
            strides[i] = 0;
            counts[i] = 1;
        }
    }

    for (size_t i = 0; i < counts[0]; i++) {
        for (size_t j = 0; j < counts[1]; j++) {
            for (size_t k = 0; k < counts[2]; k++) {
                uint64_t region_begin = begin + i * strides[0] + j * strides[1] + k * strides[2];
                uint64_t region_end = region_begin + size;
                region.Merge(ROUND_DOWN(region_begin, UVM_PAGE_SIZE),
                         ROUND_UP(region_end, UVM_PAGE_SIZE));
                // region.Merge(region_begin, region_end);
            }
        }
    }
}
