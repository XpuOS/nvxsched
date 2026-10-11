#include <cstddef>
#include <cstdint>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/hash.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/dims.h"
#include "xsched/cuda/hal/profile/relation.h"

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

std::unique_ptr<AccessRelation> AccessRelation::Create(Json::Value &relation)
{
    XASSERT(relation.isMember("type") && relation["type"].isString(), "type should be a string");
    if (relation["type"] == "EXACT_BEGIN_FIXED_SIZE") {
        return std::make_unique<ExactBeginFixedSizeRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_LINEAR_SIZE") {
        return std::make_unique<ExactBeginLinearSizeRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_MULTIPLIED_SIZE") {
        return std::make_unique<ExactBeginMultipliedSizeRelation>(relation);
    } else if (relation["type"] == "EXACT_END_FIXED_SIZE") {
        return std::make_unique<ExactEndFixedSizeRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_FIXED_SIZE_FIXED_STRIDE_FIXED_COUNT") {
        return std::make_unique<ExactBeginFixedSizeFixedStrideFixedCountRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_FIXED_SIZE_EXACT_3D_STRIDE_EXACT_3D_COUNT") {
        return std::make_unique<ExactBeginFixedSizeExact3DStrideExact3DCountRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_LINEAR_SIZE_FIXED_STRIDE_FIXED_COUNT") {
        return std::make_unique<ExactBeginLinearSizeFixedStrideFixedCountRelation>(relation);
    } else if (relation["type"] == "EXACT_BEGIN_LINEAR_SIZE_LINEAR_3D_STRIDE_LINEAR_3D_COUNT") {
        return std::make_unique<ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation>(relation);
    }

    XERRO("unknown relation type: %s", relation["type"].asString().c_str());
    return nullptr;
}

ExactBeginFixedSizeRelation::ExactBeginFixedSizeRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginFixedSize)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ = relation["size"].asUInt64();
}

ExactBeginLinearSizeRelation::ExactBeginLinearSizeRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginLinearSize)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ref_ = IntReference::Create(relation["size"]);
    size_multiplier_ = relation["size"]["multiplier"].asDouble();
}

ExactBeginMultipliedSizeRelation::ExactBeginMultipliedSizeRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginMultipliedSize)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_multiplier_ = relation["size"]["multiplier"].asDouble();
    for (auto &multiplier : relation["size"]["multipliers"]) {
        size_refs_.push_back(IntReference::Create(multiplier));
    }
}

ExactEndFixedSizeRelation::ExactEndFixedSizeRelation(Json::Value &relation)
    : AccessRelation(kRelationExactEndFixedSize)
{
    relation["end"]["size"] = sizeof(void *);
    end_ref_ = IntReference::Create(relation["end"]);
    size_ = relation["size"].asUInt64();
}

ExactBeginFixedSizeFixedStrideFixedCountRelation::
ExactBeginFixedSizeFixedStrideFixedCountRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginFixedSizeFixedStrideFixedCount)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ = relation["size"].asUInt64();
    stride_ = relation["stride"].asUInt64();
    count_ = relation["count"].asUInt64();
}

void ExactBeginFixedSizeFixedStrideFixedCountRelation::
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

ExactBeginFixedSizeExact3DStrideExact3DCountRelation::
ExactBeginFixedSizeExact3DStrideExact3DCountRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginFixedSizeExact3DStrideExact3DCount)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ = relation["size"].asUInt64();
    for (int i = 0; i < 3; i++) {
        strides_[i] = IntReference::Create(relation["stride"][i]);
        counts_[i] = IntReference::Create(relation["count"][i]);
    }
}

void ExactBeginFixedSizeExact3DStrideExact3DCountRelation::
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

ExactBeginLinearSizeFixedStrideFixedCountRelation::
ExactBeginLinearSizeFixedStrideFixedCountRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginLinearSizeFixedStrideFixedCount)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ref_ = IntReference::Create(relation["size"]);
    size_multiplier_ = relation["size"]["multiplier"].asDouble();
    stride_ = relation["stride"].asUInt64();
    count_ = relation["count"].asUInt64();
}

void ExactBeginLinearSizeFixedStrideFixedCountRelation::
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

ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation::
ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation(Json::Value &relation)
    : AccessRelation(kRelationExactBeginLinearSizeLinear3DStrideLinear3DCount)
{
    relation["begin"]["size"] = sizeof(void *);
    begin_ref_ = IntReference::Create(relation["begin"]);
    size_ref_ = IntReference::Create(relation["size"]);
    size_multiplier_ = relation["size"]["multiplier"].asDouble();
    for (int i = 0; i < 3; i++) {
        strides_[i] = IntReference::Create(relation["stride"][i]);
        counts_[i] = IntReference::Create(relation["count"][i]);
        strides_multipliers_[i] = relation["stride"][i]["multiplier"].asDouble();
        counts_multipliers_[i] = relation["count"][i]["multiplier"].asDouble();
    }
}

void ExactBeginLinearSizeLinear3DStrideLinear3DCountRelation::
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
