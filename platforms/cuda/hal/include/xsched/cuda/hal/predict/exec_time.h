#pragma once

#include <map>
#include <string>
#include <memory>
#include <unordered_map>

#include "xsched/utils/common.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/cupti.h"
#include "xsched/cuda/hal/common/kernel.h"

namespace xsched::cuda
{

#ifndef EXECUTION_TIME_INVALID
    #define EXECUTION_TIME_INVALID -1
#endif

class XExecutionTimePredictor
{
public:
    XExecutionTimePredictor() = default;
    virtual ~XExecutionTimePredictor() = default;
    virtual int64_t PredictKernel(CUcontext ctx, const KernelPredictionInfo &kernel) = 0;
    virtual int64_t PredictMemcpy(CUpti_ActivityMemcpyKind kind, size_t size) = 0;
    virtual int64_t PredictMemset(CUpti_ActivityMemoryKind kind, size_t size) = 0;
};

class OfflineExecutionTimePredictor: public XExecutionTimePredictor
{
public:
    OfflineExecutionTimePredictor(const std::string &filename);
    virtual ~OfflineExecutionTimePredictor() = default;

    virtual int64_t PredictKernel(CUcontext ctx, const KernelPredictionInfo &kernel) override;
    virtual int64_t PredictMemcpy(CUpti_ActivityMemcpyKind kind, size_t size) override;
    virtual int64_t PredictMemset(CUpti_ActivityMemoryKind kind, size_t size) override;

private:
    // kernels_[kernel][gdim][bdim] = execution time, read-only
    std::unordered_map<KernelNameHash, std::map<GridDim, std::map<BlockDim, int64_t>>> kernel_;
    // memcpy_[kind][size] = execution time, read-only
    std::map<CUpti_ActivityMemcpyKind, std::map<size_t, int64_t>> memcpy_;
    // memset_[kind][size] = execution time, read-only
    std::map<CUpti_ActivityMemoryKind, std::map<size_t, int64_t>> memset_;
};

class ExecutionTimePredictor
{
public:
    ExecutionTimePredictor();
    ~ExecutionTimePredictor() = default;

    static int64_t PredictKernel(CUcontext ctx, const KernelPredictionInfo &kernel)
    {
        return g_predictor.x_predictor->PredictKernel(ctx, kernel);
    }

    static int64_t PredictMemcpy(CUpti_ActivityMemcpyKind kind, size_t size)
    {
        return g_predictor.x_predictor->PredictMemcpy(kind, size);
    }

    static int64_t PredictMemset(CUpti_ActivityMemoryKind kind, size_t size)
    {
        return g_predictor.x_predictor->PredictMemset(kind, size);
    }

private:
    static ExecutionTimePredictor g_predictor;
    std::unique_ptr<XExecutionTimePredictor> x_predictor = nullptr;
};

} // namespace xsched::cuda
