#include <memory>
#include <vector>
#include <cstring>
#include <cuxtra/cuxtra.h>

#include "xsched/utils/log.h"
#include "xsched/utils/xassert.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/common/event_pool.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/common/cuda_command.h"
#include "xsched/cuda/hal/predict/mem_access.h"

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
    , kNameHash(GetKernelNameHash(GetContext(), kFuncHandle))
    , params_(params), extra_(extra), param_sizes_(cuXtraGetParamCount(kFuncHandle))
{
    std::vector<size_t> param_offsets(param_sizes_.size());
    for (size_t i = 0; i < param_sizes_.size(); ++i) {
        cuXtraGetParamInfo(kFuncHandle, i, &param_offsets[i], &param_sizes_[i], nullptr);
    }
    if (!deep_copy) return;

    /// @FIXME: even if param_sizes_ is empty, params or extra can be non-nullptr.
    if (param_sizes_.empty()) return;
    if (params == nullptr && extra == nullptr) {
        XWARN("params and extra of kernel (%p) are both nullptr", func);
        return;
    }
    if (params != nullptr && extra != nullptr) {
        XWARN("illegally using both params (%p) and extra (%p) of kernel %p", params, extra, func);
        return;
    }

    deep_copy_ = true;
    params_ = (void **)malloc(param_sizes_.size() * sizeof(void *));

    if (params != nullptr) {
        // Allocate a continuous buffer for all of the params
        // buffer size = last param offset + last param size
        size_t buffer_size = param_offsets.back() + param_sizes_.back();
        param_data_ = (char *)malloc(buffer_size);

        for (size_t i = 0; i < param_sizes_.size(); ++i) {
            size_t offset = param_offsets[i];
            size_t size = param_sizes_[i];
            params_[i] = (void*)&param_data_[offset];
            memcpy(params_[i], params[i], size);
        }
    } else if (extra != nullptr) {
        // Get extra buffer and its size from extra map.
        void *extra_buffer = nullptr;
        cuXtraGetExtraBuffer(extra, &extra_buffer, &extra_buffer_size_);
        // We have checked param_sizes_.size() > 0.
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
        for (size_t i = 0; i < param_sizes_.size(); ++i) {
            size_t offset = param_offsets[i];
            size_t size = param_sizes_[i];
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
    KernelPredictionInfo current {
        .func = kFuncHandle,
        .hash = kNameHash,
        .gdim = GetPackedGridDim(),
        .bdim = GetPackedBlockDim(),
        .params = params_,
        .param_sizes = param_sizes_,
    };

    // Predict execution time first.
    exec_time_ns_ = ExecutionTimePredictor::PredictKernel(GetContext(), current);

    auto prev_kern = std::move(prev_kern_);
    prev_kern_ = nullptr;

    if (xmm == nullptr) return;
    if (ctx_type == kMemoryContextUnknown)    return;
    if (ctx_type == kMemoryContextNone)       return;
    if (ctx_type == kMemoryContextAllocation) return;
    if (mem_ctx_ != nullptr) return;
    mem_ctx_ = std::make_shared<preempt::MemorySet>();

    KernelPredictionInfo previous {
        .func = prev_kern        ? prev_kern->kFuncHandle         : nullptr,
        .hash = prev_kern        ? prev_kern->kNameHash           : 0,
        .gdim = prev_kern        ? prev_kern->GetPackedGridDim()  : 0,
        .bdim = prev_kern        ? prev_kern->GetPackedBlockDim() : 0,
        .params = prev_kern      ? prev_kern->params_             : nullptr,
        .param_sizes = prev_kern ? prev_kern->param_sizes_        : std::vector<size_t>{},
    };
    MemoryAccessPredictor::PredictKernel(ctx_type, GetContext(), std::move(xmm),
                                         current, previous,
                                         mem_ctx_->reads, mem_ctx_->writes);
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

    std::unique_ptr<std::vector<void *>> prev_params = nullptr;
    auto prev_kernel = std::make_unique<KernelPredictionInfo>();
    prev_kernel->func = nullptr;
    prev_kernel->hash = 0;
    prev_kernel->gdim = 0;
    prev_kernel->bdim = 0;
    prev_kernel->params = nullptr;
    prev_kernel->param_sizes = {};

    for (size_t i = 0; i < node_cnt; ++i) {
        CUgraphNode node = nodes[i];
        CUXgraphNodeType type = cuXtraGraphNodeGetType(node);
        switch (type)
        {
        case CUX_GRAPH_NODE_TYPE_KERNEL:
        {
            CUXTRA_KERNEL_PARAMS kern {};
            cuXtraGraphKernelNodeGetParams(node, &kern);
            // kern.func could be a CUkernel.
            CUfunction func = nullptr;
            if (kern.func != nullptr) {
                func = cuXtraKernelGetFunction((CUkernel)kern.func);
            } else if (kern.kern != nullptr) {
                func = cuXtraKernelGetFunction(kern.kern);
            } else {
                XWARN("kernel node with nullptr func and kern");
                break;
            }

            auto current = std::make_unique<KernelPredictionInfo>();
            current->func = func;
            current->hash = GetKernelNameHash(GetContext(), func);
            current->gdim = PackGridDim(kern.gdim_x, kern.gdim_y, kern.gdim_z);
            current->bdim = PackBlockDim(kern.bdim_x, kern.bdim_y, kern.bdim_z);
            current->params = nullptr;
            current->param_sizes = {};

            // Predict execution time first.
            int64_t exec_time_ns = ExecutionTimePredictor::PredictKernel(GetContext(), *current);
            if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;

            if (xmm == nullptr) break;
            if (ctx_type == kMemoryContextUnknown)    break;
            if (ctx_type == kMemoryContextNone)       break;
            if (ctx_type == kMemoryContextAllocation) break;

            // Check kernel parameters.
            size_t param_cnt = cuXtraGetParamCount(func);
            if (param_cnt == 0) {
                prev_params = nullptr;
                prev_kernel = std::move(current);
                break;
            }
            XASSERT(kern.params != nullptr || kern.extra != nullptr,
                    "kernel (%p) does not have any parameters or extra", func);
            XASSERT(!(kern.params != nullptr && kern.extra != nullptr),
                    "illegally using both params (%p) and extra (%p) of kernel %p",
                    kern.params, kern.extra, func);

            // Prepare params vector.
            auto params = std::make_unique<std::vector<void *>>(param_cnt, nullptr);
            current->param_sizes.resize(param_cnt);
            if (kern.params != nullptr) {
                memcpy(params->data(), kern.params, param_cnt * sizeof(void *));
                for (size_t i = 0; i < current->param_sizes.size(); ++i) {
                    cuXtraGetParamInfo(func, i, nullptr, &current->param_sizes[i], nullptr);
                }
            } else if (kern.extra != nullptr) {
                char *extra_buffer = nullptr;
                size_t extra_buffer_size = 0;
                cuXtraGetExtraBuffer(kern.extra, (void **)&extra_buffer, &extra_buffer_size);
                XASSERT(extra_buffer != nullptr, "extra buffer of kernel (%p) is nullptr", func);
                for (size_t i = 0; i < param_cnt; ++i) {
                    size_t offset, size;
                    cuXtraGetParamInfo(func, i, &offset, &size, nullptr);
                    current->param_sizes[i] = size;
                    XASSERT(offset + size <= extra_buffer_size,
                            "extra[%zu] out of range: offset (%zu) + size (%zu) > buffer size (%zu)",
                            i, offset, size, extra_buffer_size);
                    (*params)[i] = (void*)&extra_buffer[offset];
                }
            }
            current->params = params->data();

            // Predict memory context.
            MemoryAccessPredictor::PredictKernel(ctx_type, GetContext(), xmm,
                                                 *current, *prev_kernel,
                                                 mem_ctx_->reads, mem_ctx_->writes);

            prev_params = std::move(params);
            prev_kernel = std::move(current);
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
                int64_t exec_time_ns = ExecutionTimePredictor::PredictMemcpy(
                    CUPTI_ACTIVITY_MEMCPY_KIND_HTOD, params.size);
                if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
                mem_ctx_->writes.Merge((uint64_t)params.dst, (uint64_t)params.dst + params.size);
                break;
            }
            case CUX_MEMCPY_TYPE_D2H:
            {
                int64_t exec_time_ns = ExecutionTimePredictor::PredictMemcpy(
                    CUPTI_ACTIVITY_MEMCPY_KIND_DTOH, params.size);
                if (exec_time_ns > 0) exec_time_ns_ += exec_time_ns;
                mem_ctx_->reads.Merge((uint64_t)params.src, (uint64_t)params.src + params.size);
                break;
            }
            case CUX_MEMCPY_TYPE_D2D:
            {
                int64_t exec_time_ns = ExecutionTimePredictor::PredictMemcpy(
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
            int64_t exec_time_ns = ExecutionTimePredictor::PredictMemset(
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
