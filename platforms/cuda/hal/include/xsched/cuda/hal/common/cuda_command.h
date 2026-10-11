#pragma once

#include <cstring>

#include "xsched/preempt/hal/hw_command.h"
#include "xsched/cuda/hal/common/dims.h"
#include "xsched/cuda/hal/common/cuda.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/cuda_assert.h"
#include "xsched/cuda/hal/profile/profile.h"

namespace xsched::cuda
{

#define CUDA_COMMAND(name, base, func, get_exec_time_ns, get_mem_ctx, ...) \
    class name : public base \
    { \
    public: \
        name(FOR_EACH_PAIR_COMMA(DECLARE_PARAM, __VA_ARGS__)) \
            : base(), FOR_EACH_PAIR_COMMA(DECLARE_COPY_PRIVATE_ARG, __VA_ARGS__) \
            , mem_ctx_(std::make_shared<preempt::MemorySet>()) \
        { get_mem_ctx; exec_time_ns_ = get_exec_time_ns; } \
        virtual ~name() = default; \
        virtual int64_t GetExecutionTimeNs() override { return exec_time_ns_; } \
        virtual std::shared_ptr<preempt::MemorySet> GetMemoryContext() override { return mem_ctx_; } \
    private: \
        FOR_EACH_PAIR_SEMICOLON(DECLARE_PRIVATE_PARAM, __VA_ARGS__) \
        int64_t exec_time_ns_ = -1; \
        std::shared_ptr<preempt::MemorySet> mem_ctx_ = nullptr; \
        virtual CUresult Launch(CUstream stream) override \
        { return func(FOR_EACH_PAIR_COMMA(DECLARE_PRIVATE_ARG, __VA_ARGS__), stream); } \
    };

class CudaCommand : public preempt::HwCommand
{
public:
    CudaCommand(preempt::XCommandProperties props);
    virtual ~CudaCommand();
    virtual void Synchronize() override;
    virtual bool Synchronizable() override;
    virtual bool EnableSynchronization() override;
    virtual CUresult LaunchWrapper(CUstream stream);
    CUcontext GetContext() const { return ctx_; }

private:
    CUcontext ctx_ = nullptr;
    CUevent following_event_ = nullptr;
    CUevent profile_begin_event_ = nullptr;
    CUevent profile_end_event_ = nullptr;
    virtual CUresult Launch(CUstream stream) = 0;
};

// cuda kernels
class CudaKernelCommand : public CudaCommand
{
public:
    CudaKernelCommand(CUfunction func, void **params, void **extra, bool deep_copy);
    virtual ~CudaKernelCommand();
    virtual uint64_t GetPackedGridDim() const = 0;
    virtual uint32_t GetPackedBlockDim() const = 0;

    size_t BlockCnt() const
    {
        int32_t gdx, gdy, gdz;
        UnpackGridDim(GetPackedGridDim(), gdx, gdy, gdz);
        return (size_t)gdx * (size_t)gdy * (size_t)gdz;
    }

    std::string GetKernelName() const;
    void SetPreviousKernel(std::shared_ptr<CudaKernelCommand> prev) { prev_kern_ = prev; }
    virtual int64_t GetExecutionTimeNs() override { return exec_time_ns_; }
    virtual std::shared_ptr<preempt::MemorySet> GetMemoryContext() override { return mem_ctx_; }
    virtual void Predict(XMemoryContextType ctx, std::shared_ptr<preempt::XMemoryManager> xmm) override;

    void PredictExecutionTime();
    static void PredictMemoryContextAllocation(CUfunction func, void **params, size_t param_cnt,
                                               std::shared_ptr<preempt::XMemoryManager> xmm,
                                               std::shared_ptr<preempt::MemorySet> mem_ctx);
    static bool PredictMemoryContextHeuristic(
        CUcontext ctx, CUfunction func, uint64_t pgdim, uint32_t pbdim, void **params,
        CUfunction func_prev, uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev,
        std::shared_ptr<preempt::MemorySet> mem_ctx);

    // After CUDA 12, kFunc could be a CUkernel. We convert CUkernel to CUfunction
    // by calling cuKernelGetFunction, the actual CUfunction is stored in kFuncHandle.
    const CUfunction kFunc;
    const CUfunction kFuncHandle;

    bool killable = false;
    CUdeviceptr preempt_buffer = 0;
    CUdeviceptr entry_point_original = 0;
    CUdeviceptr entry_point_guardian = 0;

protected:
    bool deep_copy_ = false;
    // If extra is passed, its param ptrs will also be filled into params_ during deep-copy.
    // When launching the kernel, if extra_ is not nullptr, use extra_ instead of params_.
    void **params_ = nullptr;
    void **extra_ = nullptr;

private:
    size_t param_cnt_ = 0;
    char *param_data_ = nullptr;
    char *extra_data_ = nullptr;
    size_t extra_buffer_size_ = 0;

    int64_t exec_time_ns_ = -1;
    // For memory context prediction.
    std::shared_ptr<preempt::MemorySet> mem_ctx_ = nullptr;
    std::shared_ptr<CudaKernelCommand> prev_kern_ = nullptr;
};

class CudaKernelLaunchCommand : public CudaKernelCommand
{
public:
    CudaKernelLaunchCommand(CUfunction func,
                            unsigned int gdx, unsigned int gdy, unsigned int gdz,
                            unsigned int bdx, unsigned int bdy, unsigned int bdz,
                            unsigned int shm, void **params, void **extra, bool deep_copy)
        : CudaKernelCommand(func, params, extra, deep_copy)
        , gdx_(gdx), gdy_(gdy), gdz_(gdz), bdx_(bdx), bdy_(bdy), bdz_(bdz), shm_(shm) {}
    virtual ~CudaKernelLaunchCommand() = default;
    virtual uint64_t GetPackedGridDim() const override { return PackGridDim(gdx_, gdy_, gdz_); }
    virtual uint32_t GetPackedBlockDim() const override { return PackBlockDim(bdx_, bdy_, bdz_); }

private:
    const unsigned int gdx_, gdy_, gdz_; // grid dimension
    const unsigned int bdx_, bdy_, bdz_; // block dimension
    const unsigned int shm_; // shared memory byte size
    virtual CUresult Launch(CUstream stream) override
    {
        void **params = params_;
        if (deep_copy_ && extra_ != nullptr) params = nullptr;
        return Driver::LaunchKernel(kFunc, gdx_, gdy_, gdz_, bdx_, bdy_, bdz_,
                                    shm_, stream, params, extra_);
    }
};

class CudaKernelLaunchExCommand : public CudaKernelCommand
{
public:
    CudaKernelLaunchExCommand(const CUlaunchConfig *cfg, CUfunction func,
                              void **params, void **extra, bool deep_copy);
    virtual ~CudaKernelLaunchExCommand();
    virtual uint64_t GetPackedGridDim() const override
    { return PackGridDim(cfg_.gridDimX, cfg_.gridDimY, cfg_.gridDimZ); }
    virtual uint32_t GetPackedBlockDim() const override
    { return PackBlockDim(cfg_.blockDimX, cfg_.blockDimY, cfg_.blockDimZ); }

private:
    CUlaunchConfig cfg_;
    virtual CUresult Launch(CUstream stream) override;
};

// host function
class CudaHostFuncCommand : public CudaCommand
{
public:
    CudaHostFuncCommand(CUhostFn fn, void *data)
        : CudaCommand(preempt::kCommandPropertyNone), fn_(fn), data_(data) {}
    virtual ~CudaHostFuncCommand() = default;

private:
    const CUhostFn fn_;
    void * const data_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::LaunchHostFunc(stream, fn_, data_); }
};

// cuda graphs
class CudaGraphCommand : public CudaCommand
{
public:
    /// @note We assue applications will not modify a CUDA graph before
    /// after submission and before waiting for its completion.
    /// Otherwise, we should do deep copy of the graph exec here,
    /// or set kCommandPropertyBlockingSubmit in the command properties.
    CudaGraphCommand(CUgraphExec graph_exec)
        : CudaCommand(preempt::kCommandPropertyNone), graph_exec_(graph_exec) {}
    virtual ~CudaGraphCommand() = default;

protected:
    CUgraphExec graph_exec_;
};

class CudaGraphUploadCommand : public CudaGraphCommand
{
public:
    CudaGraphUploadCommand(CUgraphExec graph_exec): CudaGraphCommand(graph_exec) {}
    virtual ~CudaGraphUploadCommand() = default;

private:
    virtual CUresult Launch(CUstream stream) override
    { return Driver::GraphUpload(graph_exec_, stream); }
};

class CudaGraphLaunchCommand : public CudaGraphCommand
{
public:
    CudaGraphLaunchCommand(CUgraphExec graph_exec): CudaGraphCommand(graph_exec) {}
    virtual ~CudaGraphLaunchCommand() = default;

    virtual int64_t GetExecutionTimeNs() override { return exec_time_ns_; }
    virtual std::shared_ptr<preempt::MemorySet> GetMemoryContext() override { return mem_ctx_; }
    virtual void Predict(XMemoryContextType ctx, std::shared_ptr<preempt::XMemoryManager> xmm) override;

private:
    virtual CUresult Launch(CUstream stream) override
    { return Driver::GraphLaunch(graph_exec_, stream); }

    int64_t exec_time_ns_ = -1;
    std::shared_ptr<preempt::MemorySet> mem_ctx_ = nullptr;
};

// memory commands
class CudaMemoryCommand : public CudaCommand
{
public:
    CudaMemoryCommand(): CudaCommand(preempt::kCommandPropertyNone) {}
    virtual ~CudaMemoryCommand() = default;
};

class CudaMemcpyHtoDV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpyHtoDV2Command(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
        : CudaMemoryCommand(), dstDevice_(dstDevice), srcHost_(srcHost), ByteCount_(ByteCount)
        , mem_ctx_(std::make_shared<preempt::MemorySet>())
    {
        mem_ctx_->writes.Merge(dstDevice, dstDevice + ByteCount);
        exec_time_ns_ = Profiler::GetMemcpyExecutionTimeNs(
            CUPTI_ACTIVITY_MEMCPY_KIND_HTOD, ByteCount
        );

        if (srcHost == nullptr || ByteCount == 0) return;
        // check if srcHost is pinned
        uint32_t type = 0;
        CUresult res = Driver::PointerGetAttribute(
            &type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)srcHost
        );
        if (res == CUDA_SUCCESS && type == CU_MEMORYTYPE_HOST) return; // pinned host memory
        /// FIXME: Do not copy DEVICE/ARRAY/UNIFIED memory
        void *pinned_host = nullptr;
        CUDA_ASSERT(Driver::MemAllocHost_v2(&pinned_host, ByteCount));
        srcHost_ = pinned_host;
        src_host_allocated_ = true;
        memcpy(pinned_host, srcHost, ByteCount);
    }
    virtual ~CudaMemcpyHtoDV2Command()
    {
        if (!src_host_allocated_) return;
        CUDA_ASSERT(Driver::MemFreeHost((void *)srcHost_));
    };
    virtual int64_t GetExecutionTimeNs() override { return exec_time_ns_; }
    virtual std::shared_ptr<preempt::MemorySet> GetMemoryContext() override { return mem_ctx_; }

private:
    CUdeviceptr dstDevice_;
    const void *srcHost_;
    size_t ByteCount_;
    int64_t exec_time_ns_ = -1;
    std::shared_ptr<preempt::MemorySet> mem_ctx_ = nullptr;
    bool src_host_allocated_ = false;

    virtual CUresult Launch(CUstream stream) override
    {
        return Driver::MemcpyHtoDAsync_v2(dstDevice_, srcHost_, ByteCount_, stream);
    }
};

// CUDA_COMMAND(CudaMemcpyHtoDV2Command, CudaMemoryCommand, Driver::MemcpyHtoDAsync_v2, Profiler::GetMemcpyExecutionTimeNs(CUPTI_ACTIVITY_MEMCPY_KIND_HTOD, ByteCount)                   , {mem_ctx_->writes.Merge(dstDevice, dstDevice + ByteCount);                 }, CUdeviceptr, dstDevice, const void *, srcHost, size_t, ByteCount);
CUDA_COMMAND(CudaMemcpyDtoHV2Command, CudaMemoryCommand, Driver::MemcpyDtoHAsync_v2, Profiler::GetMemcpyExecutionTimeNs(CUPTI_ACTIVITY_MEMCPY_KIND_DTOH, ByteCount)                   , {mem_ctx_->reads.Merge(srcDevice, srcDevice + ByteCount);                  }, void *, dstHost, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_COMMAND(CudaMemcpyDtoDV2Command, CudaMemoryCommand, Driver::MemcpyDtoDAsync_v2, Profiler::GetMemcpyExecutionTimeNs(CUPTI_ACTIVITY_MEMCPY_KIND_DTOD, ByteCount)                   , {mem_ctx_->reads.Merge(srcDevice, srcDevice + ByteCount); mem_ctx_->writes.Merge(dstDevice, dstDevice + ByteCount);}, CUdeviceptr, dstDevice, CUdeviceptr, srcDevice, size_t, ByteCount);
CUDA_COMMAND(CudaMemsetD8Command    , CudaMemoryCommand, Driver::MemsetD8Async     , Profiler::GetMemsetExecutionTimeNs(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, sizeof(unsigned char) * N) , {mem_ctx_->writes.Merge(dstDevice, dstDevice + N * sizeof(unsigned char)); }, CUdeviceptr, dstDevice, unsigned char, uc, size_t, N);
CUDA_COMMAND(CudaMemsetD16Command   , CudaMemoryCommand, Driver::MemsetD16Async    , Profiler::GetMemsetExecutionTimeNs(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, sizeof(unsigned short) * N), {mem_ctx_->writes.Merge(dstDevice, dstDevice + N * sizeof(unsigned short));}, CUdeviceptr, dstDevice, unsigned short, us, size_t, N);
CUDA_COMMAND(CudaMemsetD32Command   , CudaMemoryCommand, Driver::MemsetD32Async    , Profiler::GetMemsetExecutionTimeNs(CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, sizeof(unsigned int) * N)  , {mem_ctx_->writes.Merge(dstDevice, dstDevice + N * sizeof(unsigned int));  }, CUdeviceptr, dstDevice, unsigned int, ui, size_t, N);
CUDA_COMMAND(CudaMemsetD2D8Command  , CudaMemoryCommand, Driver::MemsetD2D8Async   , -1, {}, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned char, uc, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemsetD2D16Command , CudaMemoryCommand, Driver::MemsetD2D16Async  , -1, {}, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned short, us, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemsetD2D32Command , CudaMemoryCommand, Driver::MemsetD2D32Async  , -1, {}, CUdeviceptr, dstDevice, size_t, dstPitch, unsigned int, ui, size_t, Width, size_t, Height);
CUDA_COMMAND(CudaMemoryAllocCommand , CudaMemoryCommand, Driver::MemAllocAsync     , -1, {}, CUdeviceptr *, dptr, size_t, bytesize);
CUDA_COMMAND(CudaMemoryFreeCommand  , CudaMemoryCommand, Driver::MemFreeAsync      , -1, {}, CUdeviceptr, dptr);

class CudaMemcpy2DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy2DV2Command(const CUDA_MEMCPY2D *p_copy): CudaMemoryCommand(), copy_(*p_copy) {}
    virtual ~CudaMemcpy2DV2Command() = default;
private:
    const CUDA_MEMCPY2D copy_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::Memcpy2DAsync_v2(&copy_, stream); }
};

class CudaMemcpy3DV2Command : public CudaMemoryCommand
{
public:
    CudaMemcpy3DV2Command(const CUDA_MEMCPY3D *p_copy): CudaMemoryCommand(), copy_(*p_copy) {}
    virtual ~CudaMemcpy3DV2Command() = default;
private:
    const CUDA_MEMCPY3D copy_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::Memcpy3DAsync_v2(&copy_, stream); }
};

// cuda events
class CudaEventRecordCommand : public CudaCommand
{
public:
    CudaEventRecordCommand(CUevent event);
    virtual ~CudaEventRecordCommand();
    virtual void Synchronize() override { CUDA_ASSERT(Driver::EventSynchronize(event_)); }
    virtual bool Synchronizable() override { return true; }
    virtual bool EnableSynchronization() override { return true; }
    virtual CUresult LaunchWrapper(CUstream stream) override { return Launch(stream); }
    // Mark the event_ as destroyed, so that the event_ will be destroyed in the destructor.
    void DestroyEvent() { destroy_event_ = true; }

protected:
    CUevent event_;

private:
    bool destroy_event_ = false;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::EventRecord(event_, stream); }
};

class CudaEventRecordWithFlagsCommand : public CudaEventRecordCommand
{
public:
    CudaEventRecordWithFlagsCommand(CUevent event, unsigned int flags)
        : CudaEventRecordCommand(event), flags_(flags) {}
    virtual ~CudaEventRecordWithFlagsCommand() = default;

private:
    const unsigned int flags_;
    virtual CUresult Launch(CUstream stream) override
    { return Driver::EventRecordWithFlags(event_, stream, flags_); }
};

class CudaEventWaitCommand : public CudaCommand
{
public:
    CudaEventWaitCommand(CUevent event, unsigned int flags)
        : CudaCommand(preempt::kCommandPropertyIdempotent)
        , event_(event), event_cmd_(nullptr), flags_(flags) {}
    CudaEventWaitCommand(std::shared_ptr<CudaEventRecordCommand> event_cmd, unsigned int flags)
        : CudaCommand(preempt::kCommandPropertyIdempotent)
        , event_(nullptr), event_cmd_(event_cmd), flags_(flags) {}
    virtual ~CudaEventWaitCommand() = default;
    virtual void BeforeLaunch() override;

private:
    const CUevent event_; // the event to wait is recorded on a normal cuda stream
    const std::shared_ptr<CudaEventRecordCommand> event_cmd_; // recorded on an XQueue
    const unsigned int flags_;
    virtual CUresult Launch(CUstream stream) override;
};

} // namespace xsched::cuda
