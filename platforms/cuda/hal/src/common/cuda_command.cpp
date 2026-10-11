#include <memory>
#include <vector>
#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/cuda/hal/common/dims.h"
#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/common/event_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cuda_command.h"

using namespace xsched::cuda;

CudaCommand::CudaCommand(preempt::XCommandProperties props)
    : HwCommand(GetCudaLv3Implementation() == kCudaLv3ImplementationTsg
                ? props | preempt::kCommandPropertyDeactivatable : props)
{
    CUDA_ASSERT(Driver::CtxGetCurrent(&ctx_));
    XASSERT(ctx_ != nullptr, "current context of the calling thread is nullptr");
}

CudaCommand::~CudaCommand()
{
    if (following_event_ != nullptr) CudaEventPool::Push(ctx_, following_event_);
    if (profile_begin_event_ != nullptr) {
        CUDA_ASSERT(Driver::EventDestroy_v2(profile_begin_event_));
    }
    if (profile_end_event_ != nullptr) {
        CUDA_ASSERT(Driver::EventDestroy_v2(profile_end_event_));
    }
}

void CudaCommand::Synchronize()
{
    XASSERT(following_event_ != nullptr,
            "following_event_ is nullptr, EnableSynchronization() should be called first");
    CUDA_ASSERT(Driver::EventSynchronize(following_event_));
}

bool CudaCommand::Synchronizable()
{
    return following_event_ != nullptr;
}

bool CudaCommand::EnableSynchronization()
{
    if (following_event_ != nullptr) return true;
    following_event_ = CudaEventPool::Pop(ctx_);
    return following_event_ != nullptr;
}

CUresult CudaCommand::LaunchWrapper(CUstream stream)
{
    CUresult ret = CUDA_SUCCESS;

    if (profile_begin_event_ != nullptr) {
        ret = Driver::EventRecord(profile_begin_event_, stream);
        if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;
    }

    ret = Launch(stream);
    if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;

    if (profile_end_event_ != nullptr) {
        ret = Driver::EventRecord(profile_end_event_, stream);
        if (UNLIKELY(ret != CUDA_SUCCESS)) return ret;
    }

    if (following_event_ != nullptr) {
        ret = Driver::EventRecord(following_event_, stream);
    }
    return ret;
}

CudaKernelCommand::CudaKernelCommand(CUfunction func, void **params, void **extra, bool deep_copy)
    : CudaCommand(preempt::kCommandPropertyDeactivatable)
    // cuXtraKernelGetFunction() will convert func to CUfunction if func is a CUkernel.
    , kFunc(func), kFuncHandle(cuXtraKernelGetFunction((CUkernel)func))
    , params_(params), extra_(extra), param_cnt_(cuXtraGetParamCount(kFuncHandle))
{
    if (!deep_copy) return;

    /// @FIXME: even if param_cnt_ == 0, params or extra can be non-nullptr.
    if (param_cnt_ == 0) return;
    if (params == nullptr && extra == nullptr) {
        XWARN("params and extra of kernel (%p) are both nullptr", func);
        return;
    }
    if (params != nullptr && extra != nullptr) {
        XWARN("illegally using both params (%p) and extra (%p) of kernel %p", params, extra, func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_cnt_ * sizeof(void *));

    if (params != nullptr) {
        // Allocate a continuous buffer for all of the params
        // buffer size = last param offset + last param size
        size_t last_offset, last_size;
        cuXtraGetParamInfo(kFuncHandle, param_cnt_ - 1, &last_offset, &last_size, nullptr);
        size_t buffer_size = last_offset + last_size;
        param_data_ = (char *)malloc(buffer_size);

        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset, size;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            params_[i] = (void*)&param_data_[offset];
            memcpy(params_[i], params[i], size);
        }
    } else if (extra != nullptr) {
        // Get extra buffer and its size from extra map.
        void *extra_buffer = nullptr;
        cuXtraGetExtraBuffer(extra, &extra_buffer, &extra_buffer_size_);
        // We have checked param_cnt_ > 0.
        XASSERT(extra_buffer != nullptr && extra_buffer_size_ > 0,
                "invalid extra buffer (%p) and size (%zu)", extra_buffer, extra_buffer_size_);

        // Deep-copy extra buffer to extra_data_.
        extra_data_ = (char *)malloc(extra_buffer_size_);
        memcpy(extra_data_, extra_buffer, extra_buffer_size_);

        // Set extra map.
        extra_ = (void **)malloc(5 * sizeof(void *));
        extra_[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        extra_[1] = extra_data_;
        extra_[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        extra_[3] = (void *)&extra_buffer_size_;
        extra_[4] = CU_LAUNCH_PARAM_END;

        // Set params_ to point to extra_data_.
        for (size_t i = 0; i < param_cnt_; ++i) {
            size_t offset, size;
            cuXtraGetParamInfo(kFuncHandle, i, &offset, &size, nullptr);
            XASSERT(offset + size <= extra_buffer_size_,
                    "extra[%zu] out of range: offset (%zu) + size (%zu) > buffer size (%zu)",
                    i, offset, size, extra_buffer_size_);
            params_[i] = (void*)&extra_data_[offset];
        }
    }
}

CudaKernelCommand::~CudaKernelCommand()
{
    if (!deep_copy_) return;
    if (param_data_) free(param_data_);
    if (extra_data_) free(extra_data_);
    if (params_) free(params_);
    if (extra_) free(extra_);
}

std::string CudaKernelCommand::GetKernelName() const
{
    const char *name = nullptr;
    cuXtraGetFunctionName(GetContext(), kFuncHandle, &name);
    return std::string(name);
}

void CudaKernelCommand::Predict(XMemoryContextType ctx_type, std::shared_ptr<preempt::XMemoryManager> xmm)
{
    // Predict execution time first.
    this->PredictExecutionTime();

    auto prev_kern = std::move(prev_kern_);
    prev_kern_ = nullptr;

    if (xmm == nullptr) return;
    if (ctx_type <= kMemoryContextFull) return; // Unknown, NONE or FULL
    if (mem_ctx_ != nullptr) return;
    mem_ctx_ = std::make_shared<preempt::MemorySet>();

    switch (ctx_type) {
    case kMemoryContextAllocation:
        PredictMemoryContextAllocation(kFuncHandle, params_, param_cnt_, std::move(xmm), mem_ctx_);
        break;
    case kMemoryContextHeuristic:
    case kMemoryContextHeuristicAllocation:
    {
        CUfunction func_prev = prev_kern ? prev_kern->kFuncHandle : nullptr;
        uint64_t pgdim_prev = prev_kern ? prev_kern->GetPackedGridDim() : 0;
        uint32_t pbdim_prev = prev_kern ? prev_kern->GetPackedBlockDim() : 0;
        void **params_prev = prev_kern ? prev_kern->params_ : nullptr;

        bool predicted = PredictMemoryContextHeuristic(this->GetContext(),
            kFuncHandle, this->GetPackedGridDim(), this->GetPackedBlockDim(), params_,
            func_prev, pgdim_prev, pbdim_prev, params_prev, mem_ctx_);
        if (ctx_type == kMemoryContextHeuristic || predicted) break;
        // If failed to predict memory context based on heuristic rules,
        // use allocation-based prediction instead.
        PredictMemoryContextAllocation(kFuncHandle, params_, param_cnt_, std::move(xmm), mem_ctx_);
        break;
    }
    default:
        XERRO("invalid memory context type: %d", ctx_type);
        break;
    };
}

void CudaKernelCommand::PredictExecutionTime()
{
    int32_t gdx, gdy, gdz;
    int32_t bdx, bdy, bdz;
    UnpackGridDim(GetPackedGridDim(), gdx, gdy, gdz);
    UnpackBlockDim(GetPackedBlockDim(), bdx, bdy, bdz);
    exec_time_ns_ = Profiler::GetKernelExecutionTimeNs(
        GetContext(), kFuncHandle, gdx, gdy, gdz, bdx, bdy, bdz);
    if (exec_time_ns_ > 0) return;
    XDEBG("failed to predict execution time of kernel gdim[%d, %d, %d] bdim[%d, %d, %d] %s",
          gdx, gdy, gdz, bdx, bdy, bdz, GetKernelName().c_str());
}

void CudaKernelCommand::PredictMemoryContextAllocation(
    CUfunction func, void **params, size_t param_cnt,
    std::shared_ptr<preempt::XMemoryManager> xmm,
    std::shared_ptr<preempt::MemorySet> mem_ctx)
{
    for (size_t i = 0; i < param_cnt; ++i) {
        size_t param_size = 0;
        cuXtraGetParamInfo(func, i, nullptr, &param_size, nullptr);
        for (size_t offset = 0; offset + sizeof(void *) <= param_size; offset += sizeof(void *)) {
            // Scan all pointer parameters.
            void *ptr = *(void **)(uint64_t(params[i]) + offset);
            size_t size = xmm->GetAllocatedSize(ptr);
            if (size == 0) continue; // Not an allocated buffer.
            mem_ctx->reads.Merge((uint64_t)ptr, (uint64_t)ptr + size);
            mem_ctx->writes.Merge((uint64_t)ptr, (uint64_t)ptr + size);
        }
    }
}

bool CudaKernelCommand::PredictMemoryContextHeuristic(
    CUcontext ctx, CUfunction func, uint64_t pgdim, uint32_t pbdim, void **params,
    CUfunction func_prev, uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev,
    std::shared_ptr<preempt::MemorySet> mem_ctx)
{
    bool found = Profiler::GetKernelMemAccess(mem_ctx->reads, mem_ctx->writes, ctx,
        func, pgdim, pbdim, params, func_prev, pgdim_prev, pbdim_prev, params_prev);

    if (found) return true;
    int32_t gdx, gdy, gdz;
    int32_t bdx, bdy, bdz;
    UnpackGridDim(pgdim, gdx, gdy, gdz);
    UnpackBlockDim(pbdim, bdx, bdy, bdz);
    const char *name = nullptr;
    cuXtraGetFunctionName(ctx, func, &name);
    XDEBG("failed to predict memory context (heuristic) of kernel "
          "gdim[%d, %d, %d] bdim[%d, %d, %d] %s", gdx, gdy, gdz, bdx, bdy, bdz, name);
    return false;
}

CudaKernelLaunchExCommand::CudaKernelLaunchExCommand(const CUlaunchConfig *cfg, CUfunction func,
                                                     void **params, void **extra, bool deep_copy)
    : CudaKernelCommand(func, params, extra, deep_copy)
{
    if (cfg == nullptr) {
        XWARN("CUlaunchConfig of %p is nullptr", func);
        return;
    }

    memcpy(&cfg_, cfg, sizeof(CUlaunchConfig));
    if (cfg->attrs == nullptr || !deep_copy_) return;
    cfg_.attrs = (CUlaunchAttribute *)malloc(cfg->numAttrs * sizeof(CUlaunchAttribute));
    memcpy(cfg_.attrs, cfg->attrs, cfg->numAttrs * sizeof(CUlaunchAttribute));
}

CudaKernelLaunchExCommand::~CudaKernelLaunchExCommand()
{
    if (cfg_.attrs == nullptr || !deep_copy_) return;
    free(cfg_.attrs);
}

CUresult CudaKernelLaunchExCommand::Launch(CUstream stream)
{
    cfg_.hStream = stream;
    void **params = params_;
    if (deep_copy_ && extra_ != nullptr) params = nullptr;
    return Driver::LaunchKernelEx(&cfg_, kFunc, params, extra_);
}

void CudaGraphLaunchCommand::
Predict(XMemoryContextType ctx_type, std::shared_ptr<preempt::XMemoryManager> xmm)
{
    size_t node_cnt = 0;
    cuXtraGraphExecGetNodes(graph_exec_, 0, nullptr, &node_cnt);
    std::vector<CUgraphNode> nodes(node_cnt);
    cuXtraGraphExecGetNodes(graph_exec_, node_cnt, nodes.data(), &node_cnt);

    exec_time_ns_ = 0;
    mem_ctx_ = std::make_shared<preempt::MemorySet>();

    CUfunction prev_func = nullptr;
    std::unique_ptr<std::vector<void *>> prev_params = nullptr;
    std::unique_ptr<CUXTRA_KERNEL_PARAMS> prev_kern = nullptr;

    for (size_t i = 0; i < node_cnt; ++i) {
        CUgraphNode node = nodes[i];
        CUXgraphNodeType type = cuXtraGraphNodeGetType(node);
        switch (type)
        {
        case CUX_GRAPH_NODE_TYPE_KERNEL:
        {
            std::unique_ptr<CUXTRA_KERNEL_PARAMS> kern = std::make_unique<CUXTRA_KERNEL_PARAMS>();
            cuXtraGraphKernelNodeGetParams(node, kern.get());
            // kern->func could be a CUkernel.
            CUfunction func = nullptr;
            if (kern->func != nullptr) {
                func = cuXtraKernelGetFunction((CUkernel)kern->func);
            } else if (kern->kern != nullptr) {
                func = cuXtraKernelGetFunction(kern->kern);
            } else {
                XWARN("kernel node with nullptr func and kern");
                break;
            }

            // Predict execution time first.
            int64_t exec_time_ns = Profiler::GetKernelExecutionTimeNs(
                GetContext(), func, kern->gdim_x, kern->gdim_y, kern->gdim_z,
                kern->bdim_x, kern->bdim_y, kern->bdim_z);
            if (exec_time_ns <= 0) {
                const char *name = nullptr;
                cuXtraGetFunctionName(GetContext(), func, &name);
                XDEBG("failed to predict execution time of kernel gdim[%d, %d, %d] bdim[%d, %d, %d] %s",
                      kern->gdim_x, kern->gdim_y, kern->gdim_z,
                      kern->bdim_x, kern->bdim_y, kern->bdim_z, name);
            } else {
                exec_time_ns_ += exec_time_ns;
            }

            // ctx_type == Unknown, NONE or FULL
            if (xmm == nullptr || ctx_type <= kMemoryContextFull) break;

            // Check kernel parameters.
            size_t param_cnt = cuXtraGetParamCount(func);
            if (param_cnt == 0) {
                prev_func = func;
                prev_params = nullptr;
                prev_kern = std::move(kern);
                break;
            }
            XASSERT(kern->params != nullptr || kern->extra != nullptr,
                "kernel (%p) does not have any parameters or extra", func);
            XASSERT(!(kern->params != nullptr && kern->extra != nullptr),
                    "illegally using both params (%p) and extra (%p) of kernel %p",
                    kern->params, kern->extra, func);

            // Prepare params vector.
            auto params = std::make_unique<std::vector<void *>>(param_cnt, nullptr);
            if (kern->params != nullptr) {
                memcpy(params->data(), kern->params, param_cnt * sizeof(void *));
            } else if (kern->extra != nullptr) {
                char *extra_buffer = nullptr;
                size_t extra_buffer_size = 0;
                cuXtraGetExtraBuffer(kern->extra, (void **)&extra_buffer, &extra_buffer_size);
                XASSERT(extra_buffer != nullptr, "extra buffer of kernel (%p) is nullptr", func);
                for (size_t i = 0; i < param_cnt; ++i) {
                    size_t offset, size;
                    cuXtraGetParamInfo(func, i, &offset, &size, nullptr);
                    XASSERT(offset + size <= extra_buffer_size,
                            "extra[%zu] out of range: offset (%zu) + size (%zu) > buffer size (%zu)",
                            i, offset, size, extra_buffer_size);
                    (*params)[i] = (void*)&extra_buffer[offset];
                }
            }

            // Predict memory context.
            switch (ctx_type)
            {
            case kMemoryContextAllocation:
                CudaKernelCommand::PredictMemoryContextAllocation(
                    func, params->data(), cuXtraGetParamCount(func), xmm, mem_ctx_);
                break;
            case kMemoryContextHeuristic:
            case kMemoryContextHeuristicAllocation:
            {
                uint64_t prev_pgdim = prev_kern == nullptr ? 0
                    : PackGridDim(prev_kern->gdim_x, prev_kern->gdim_y, prev_kern->gdim_z);
                uint32_t prev_pbdim = prev_kern == nullptr ? 0
                    : PackBlockDim(prev_kern->bdim_x, prev_kern->bdim_y, prev_kern->bdim_z);
                void **prev_params_raw = prev_params == nullptr ? nullptr : prev_params->data();

                bool predicted = CudaKernelCommand::PredictMemoryContextHeuristic(
                    GetContext(), func,
                    PackGridDim(kern->gdim_x, kern->gdim_y, kern->gdim_z),
                    PackBlockDim(kern->bdim_x, kern->bdim_y, kern->bdim_z),
                    params->data(), prev_func, prev_pgdim, prev_pbdim, prev_params_raw,
                    mem_ctx_);
                if (ctx_type == kMemoryContextHeuristicAllocation && !predicted) {
                    // If failed to predict memory context based on heuristic rules,
                    // use allocation-based prediction instead.
                    CudaKernelCommand::PredictMemoryContextAllocation(
                        func, params->data(), cuXtraGetParamCount(func), xmm, mem_ctx_);
                }

                prev_func = func;
                prev_params = std::move(params);
                prev_kern = std::move(kern);
                break;
            }
            default:
                XERRO("invalid memory context type: %d", ctx_type);
                break;
            };
            break;
        }
        case CUX_GRAPH_NODE_TYPE_MEMCPY:
        {
            CUXTRA_MEMCPY_PARAMS params;
            cuXtraGraphMemcpyNodeGetParams(node, &params);
            switch (params.type)
            {
            case CUX_MEMCPY_TYPE_H2D:
            {
                int64_t exec_time_ns = Profiler::GetMemcpyExecutionTimeNs(
                    CUPTI_ACTIVITY_MEMCPY_KIND_HTOD, params.size);
                if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
                mem_ctx_->writes.Merge((uint64_t)params.dst, (uint64_t)params.dst + params.size);
                break;
            }
            case CUX_MEMCPY_TYPE_D2H:
            {
                int64_t exec_time_ns = Profiler::GetMemcpyExecutionTimeNs(
                    CUPTI_ACTIVITY_MEMCPY_KIND_DTOH, params.size);
                if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
                mem_ctx_->reads.Merge((uint64_t)params.src, (uint64_t)params.src + params.size);
                break;
            }
            case CUX_MEMCPY_TYPE_D2D:
            {
                int64_t exec_time_ns = Profiler::GetMemcpyExecutionTimeNs(
                    CUPTI_ACTIVITY_MEMCPY_KIND_DTOD, params.size);
                if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
                mem_ctx_->reads.Merge((uint64_t)params.src, (uint64_t)params.src + params.size);
                mem_ctx_->writes.Merge((uint64_t)params.dst, (uint64_t)params.dst + params.size);
                break;
            }
            default:
                XERRO("invalid cux memcpy type: %d", params.type);
                break;
            }
            break;
        }
        case CUX_GRAPH_NODE_TYPE_MEMSET:
        {
            CUXTRA_MEMSET_PARAMS params;
            cuXtraGraphMemsetNodeGetParams(node, &params);
            size_t total_bytes = params.elem_size * params.width * params.height;
            int64_t exec_time_ns = Profiler::GetMemsetExecutionTimeNs(
                CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, total_bytes);
            if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
            mem_ctx_->writes.Merge((uint64_t)params.dst, (uint64_t)params.dst + total_bytes);
            break;
        }
        default:
            XERRO("invalid cux graph node type: %d", type);
            break;
        }
    }
}

CudaEventRecordCommand::CudaEventRecordCommand(CUevent event)
    : CudaCommand(preempt::kCommandPropertyIdempotent), event_(event)
{
    XASSERT(event_ != nullptr, "cuda event should not be nullptr");
}

CudaEventRecordCommand::~CudaEventRecordCommand()
{
    if (event_ == nullptr || (!destroy_event_)) return;
    CUDA_ASSERT(Driver::EventDestroy_v2(event_));
}

void CudaEventWaitCommand::BeforeLaunch()
{
    if (event_cmd_) event_cmd_->Wait(); // recorded on an XQueue, wait it on the XQueue
}

CUresult CudaEventWaitCommand::Launch(CUstream stream)
{
    if (!event_) return CUDA_SUCCESS; // already waited in BeforeLaunch()
    return Driver::StreamWaitEvent(stream, event_, flags_);
}
