#include <fstream>
#include <json/json.h>
#include <cuxtra/cuxtra.h>

#include "xsched/preempt/xmem/xmem.h"
#include "xsched/cuda/hal/predict/exec_time.h"

using namespace xsched::cuda;

OfflineExecutionTimePredictor::OfflineExecutionTimePredictor(const std::string &filename)
{
    Json::Reader reader;
    std::ifstream file(filename);
    Json::Value exec_times;
    reader.parse(file, exec_times);
    file.close();

    for (auto &obj : exec_times) {
        if (obj.isMember("kernel")) {
            KernelNameHash hash = obj["kernel"].asUInt64();
            auto &gdim = obj["gdim"];
            auto &bdim = obj["bdim"];
            XASSERT(gdim.size() == 3, "gdim size should be 3");
            XASSERT(bdim.size() == 3, "bdim size should be 3");
            GridDim gdim_pack = PackGridDim(gdim[0].asInt(), gdim[1].asInt(), gdim[2].asInt());
            BlockDim bdim_pack = PackBlockDim(bdim[0].asInt(), bdim[1].asInt(), bdim[2].asInt());
            int64_t time_ns = obj["median"].asInt64();
            kernel_[hash][gdim_pack][bdim_pack] = time_ns;
        } else if (obj.isMember("memcpy")) {
            CUpti_ActivityMemcpyKind kind = (CUpti_ActivityMemcpyKind)obj["memcpy"].asInt();
            uint64_t size = obj["size"].asUInt64();
            int64_t time_ns = obj["median"].asInt64();
            memcpy_[kind][size] = time_ns;
        } else if (obj.isMember("memset")) {
            CUpti_ActivityMemoryKind kind = (CUpti_ActivityMemoryKind)obj["memset"].asInt();
            uint64_t size = obj["size"].asUInt64();
            int64_t time_ns = obj["median"].asInt64();
            memset_[kind][size] = time_ns;
        }
    }
}

int64_t OfflineExecutionTimePredictor::PredictKernel(CUcontext ctx, const KernelPredictionInfo &kernel)
{
    if (kernel.hash != 0) {
        auto kit = kernel_.find(kernel.hash);
        if (kit != kernel_.end()) {
            auto git = kit->second.find(kernel.gdim);
            if (git != kit->second.end()) {
                auto bit = git->second.find(kernel.bdim);
                if (bit != git->second.end()) {
                    return bit->second;
                }
            }
        }
    }

    if (!DebugLogEnabled()) return EXECUTION_TIME_INVALID;
    int32_t gdx, gdy, gdz;
    int32_t bdx, bdy, bdz;
    UnpackGridDim(kernel.gdim, gdx, gdy, gdz);
    UnpackBlockDim(kernel.bdim, bdx, bdy, bdz);
    const char *name = nullptr;
    cuXtraGetFunctionName(ctx, kernel.func, &name);
    XDEBG("failed to predict execution time of kernel gdim[%d, %d, %d] bdim[%d, %d, %d] %s",
          gdx, gdy, gdz, bdx, bdy, bdz, name);
    return EXECUTION_TIME_INVALID;
}

int64_t OfflineExecutionTimePredictor::PredictMemcpy(CUpti_ActivityMemcpyKind kind, size_t size)
{
    auto kit = memcpy_.find(kind);
    if (kit == memcpy_.end()) return EXECUTION_TIME_INVALID;

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) return EXECUTION_TIME_INVALID;
    return sit->second;
}

int64_t OfflineExecutionTimePredictor::PredictMemset(CUpti_ActivityMemoryKind kind, size_t size)
{
    auto kit = memset_.find(kind);
    if (kit == memset_.end()) return EXECUTION_TIME_INVALID;

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) return EXECUTION_TIME_INVALID;
    return sit->second;
}

ExecutionTimePredictor ExecutionTimePredictor::g_predictor;

ExecutionTimePredictor::ExecutionTimePredictor()
{
    /// @TODO: other prediction methods, such as online prediction using CUPTI
    const std::string filename = preempt::GetExecutionTimeFileName();
    x_predictor = std::make_unique<OfflineExecutionTimePredictor>(filename);
}
