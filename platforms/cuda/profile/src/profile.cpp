#include <map>
#include <list>
#include <cmath>
#include <mutex>
#include <vector>
#include <atomic>
#include <fstream>
#include <cstdint>
#include <algorithm>

#include "xsched/utils/log.h"
#include "xsched/utils/hash.h"
#include "xsched/utils/common.h"
#include "xsched/preempt/xmem/xmem.h"
#include "xsched/cuda/profile/profile.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

namespace xsched::cuda
{

const size_t kWarmupCnt = GetWarmupCnt();
const size_t kProfileCnt = GetProfileCnt();

CUresult XInit(unsigned int Flags)
{
    XASSERT(kWarmupCnt < kProfileCnt, "kWarmupCnt (%lu) must be less than kProfileCnt (%lu)",
            kWarmupCnt, kProfileCnt);
    PtiInit();
    return Driver::Init(Flags);
}

void CUPTIAPI PtiBufferRequested(uint8_t **buffer, size_t *size, size_t *max_num_records)
{
#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) (((uintptr_t)(buffer) & ((align)-1)) \
        ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

    *size = BUF_SIZE;
    *buffer = (uint8_t*)malloc(BUF_SIZE + ALIGN_SIZE);
    *buffer = (uint8_t*)ALIGN_BUFFER(*buffer, ALIGN_SIZE);
    *max_num_records = 0; // no limit
}

void CUPTIAPI PtiBufferCompleted(CUcontext ctx, uint32_t stream_id,
                                 uint8_t *buffer, size_t size, size_t valid_size)
{
    UNUSED(size);
    static std::mutex mtx;
    CUptiResult status;
    CUpti_Activity *record = nullptr;

    while (true) {
        status = PTI::ActivityGetNextRecord(buffer, valid_size, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        CUPTI_ASSERT(status);

        std::lock_guard<std::mutex> lock(mtx);
        switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
            PtiRecvKernelRecord((CUpti_ActivityKernel6 *)record);
            break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
            PtiRecvMemcpyRecord((CUpti_ActivityMemcpy6 *)record);
            break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
            PtiRecvMemsetRecord((CUpti_ActivityMemset4 *)record);
            break;
        default: break;
        }
    }

    size_t dropped;
    CUPTI_ASSERT(PTI::ActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
    if (dropped != 0) XWARN("dropped %lu activity records", dropped);

    free(buffer);
}

void PtiInit()
{
    static std::atomic_bool initialized = false;
    if (initialized.exchange(true)) return;

    // All variables initialized after atexit(PtiFini)
    // is destroyed before PtiFini() is called.
    // So we can call PtiSave() here to make sure
    // all static variables are initialized before atexit(PtiFini).
    PtiSave();
    atexit(PtiFini);
    CUPTI_ASSERT(PTI::ActivityRegisterCallbacks(PtiBufferRequested, PtiBufferCompleted));
    CUPTI_ASSERT(PTI::ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_ASSERT(PTI::ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_ASSERT(PTI::ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    XINFO("cupti initialized");
}

template <typename T>
double GetMedian(std::vector<T> sorted)
{
    if (sorted.empty()) return 0.0;
    size_t n = sorted.size();
    return (n % 2) ? double(sorted[n/2]) : 0.5 * double(sorted[n/2 - 1] + sorted[n/2]);
}

double MADBasedMedian(const std::list<int64_t>::iterator begin, const std::list<int64_t>::iterator end)
{
    if (begin == end) return 0.0;
    // Convert to sorted vector for median calculations
    std::vector<int64_t> vals(begin, end);
    std::sort(vals.begin(), vals.end());
    double median = GetMedian(vals);
    // Compute absolute deviations from the median
    std::vector<double> abs_dev;
    abs_dev.reserve(vals.size());
    for (int64_t x : vals) abs_dev.push_back(std::abs(double(x) - median));
    std::sort(abs_dev.begin(), abs_dev.end());
    // Compute MAD (median of absolute deviations)
    double MAD = GetMedian(abs_dev);
    if (MAD == 0.0) {
        // Data is too concentrated, return median directly
        return median;
    }
    // Filter values using modified Z-score
    std::vector<int64_t> filtered;
    for (int64_t x : vals) {
        // 0.6745: 75th percentile of the standard normal distribution
        double Mi = 0.6745 * (double(x) - median) / MAD;
        // 3.5: 3-sigma rule
        if (std::abs(Mi) <= 3.5) filtered.push_back(x);
    }
    // Return median of filtered values
    if (!filtered.empty()) {
        std::sort(filtered.begin(), filtered.end());
        return GetMedian(filtered);
    } else {
        // If all values are removed, fall back to original median
        return median;
    }
}

std::map<const char *, uint64_t> name_hash;
std::map<uint64_t, std::string> kernel_name;
// kernel_history[kernel][gdim][bdim] = duration[]
std::map<uint64_t, std::map<uint64_t, std::map<uint32_t, std::list<int64_t>>>> kernel_history;
// memcpy_history[kind][size] = duration[]
std::map<uint8_t, std::map<size_t, std::list<int64_t>>> memcpy_history;
// memset_history[kind][size] = duration[]
std::map<uint16_t, std::map<size_t, std::list<int64_t>>> memset_history;


std::map<uint8_t, std::string> memcpy_type_name = {
    {CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN, "UNK"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_HTOD, "H2D"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_DTOH, "D2H"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_HTOA, "H2A"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_ATOH, "A2H"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_ATOA, "A2A"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_ATOD, "A2D"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_DTOA, "D2A"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_DTOD, "D2D"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_HTOH, "H2H"},
    {CUPTI_ACTIVITY_MEMCPY_KIND_PTOP, "P2P"},
};

std::map<uint8_t, std::string> memset_type_name = {
    {CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN, "UNK"},
    {CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE, "PAGEABLE"},
    {CUPTI_ACTIVITY_MEMORY_KIND_PINNED, "PINNED"},
    {CUPTI_ACTIVITY_MEMORY_KIND_DEVICE, "DEVICE"},
    {CUPTI_ACTIVITY_MEMORY_KIND_ARRAY, "ARRAY"},
    {CUPTI_ACTIVITY_MEMORY_KIND_MANAGED, "MANAGED"},
    {CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC, "DEVICE_STATIC"},
    {CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC, "MANAGED_STATIC"},
};

void PtiFini()
{
    PTI::ActivityFlushAll(0);
    PtiSave();

    for (auto &[hash, gmap] : kernel_history) {
        auto kit = kernel_name.find(hash);
        if (kit == kernel_name.end()) continue;
        std::string name = kit->second;
        for (auto &[gdim, bmap] : gmap) {
            int32_t gx, gy, gz;
            UnpackGridDim(gdim, gx, gy, gz);
            for (auto &[bdim, duration] : bmap) {
                if (duration.size() < kProfileCnt) {
                    int32_t bx, by, bz;
                    UnpackBlockDim(bdim, bx, by, bz);
                    XWARN("profile cnt insufficient (%lu/%lu) for kernel %s"
                          "gdim[%u,%u,%u], bdim[%u,%u,%u]",
                          duration.size(), kProfileCnt, name.c_str(),
                          gx, gy, gz, bx, by, bz);
                }
            }
        }
    }
    for (auto &[kind, smap] : memcpy_history) {
        for (auto &[size, duration] : smap) {
            if (duration.size() < kProfileCnt) {
                XWARN("profile cnt insufficient (%lu/%lu) for memcpy kind %u, size %lu",
                      duration.size(), kProfileCnt, kind, size);
            }
        }
    }
    for (auto &[kind, smap] : memset_history) {
        for (auto &[size, duration] : smap) {
            if (duration.size() < kProfileCnt) {
                XWARN("profile cnt insufficient (%lu/%lu) for memset kind %u, size %lu",
                      duration.size(), kProfileCnt, kind, size);
            }
        }
    }
    XINFO("cupti finalized");
}

void PtiSave()
{
    std::ofstream ofs(preempt::GetExecutionTimeFileName());
    ofs << "[";
    bool first = true;

    for (auto &[hash, gmap] : kernel_history) {
        auto kit = kernel_name.find(hash);
        if (kit == kernel_name.end()) continue;
        std::string name = kit->second;
        for (auto &[gdim, bmap] : gmap) {
            int32_t gx, gy, gz;
            UnpackGridDim(gdim, gx, gy, gz);
            for (auto &[bdim, duration] : bmap) {
                int32_t bx, by, bz;
                UnpackBlockDim(bdim, bx, by, bz);
                if (duration.size() < kProfileCnt) {
                    XDEBG("profile cnt of kernel %s: %lu, require %lu",
                          name.c_str(), duration.size(), kProfileCnt);
                    continue;
                }
                auto dit = duration.begin();
                for (size_t w = 0; w < kWarmupCnt; w++) dit++;
                uint64_t median = MADBasedMedian(dit, duration.end());
                if (first) {
                    ofs << "\n";
                    first = false;
                } else {
                    ofs << ",\n";
                }
                ofs << "\t{\"kernel\":" << hash
                    << ",\"gdim\":[" << gx << "," << gy << "," << gz
                    << "],\"bdim\":[" << bx << "," << by << "," << bz
                    << "],\"median\":" << median
                    << ",\"cnt\":" << duration.size()
                    << ",\"name\":\"" << name << "\"}";
            }
        }
    }

    for (auto &[kind, smap] : memcpy_history) {
        for (auto &[size, duration] : smap) {
            if (duration.size() < kProfileCnt) {
                XDEBG("profile cnt of memcpy kind %u: %lu, require %lu",
                      kind, duration.size(), kProfileCnt);
                continue;
            }
            auto dit = duration.begin();
            for (size_t w = 0; w < kWarmupCnt; w++) dit++;
            uint64_t median = MADBasedMedian(dit, duration.end());
            if (first) {
                ofs << "\n";
                first = false;
            } else {
                ofs << ",\n";
            }
            ofs << "\t{\"memcpy\":" << (int)kind
                << ",\"type\":\"" << memcpy_type_name[kind] << "\""
                << ",\"size\":" << size
                << ",\"median\":" << median
                << ",\"cnt\":" << duration.size()
                << "}";
        }
    }

    for (auto &[kind, smap] : memset_history) {
        for (auto &[size, duration] : smap) {
            if (duration.size() < kProfileCnt) {
                XDEBG("profile cnt of memcpy kind %u: %lu, require %lu",
                      kind, duration.size(), kProfileCnt);
                continue;
            }
            auto dit = duration.begin();
            for (size_t w = 0; w < kWarmupCnt; w++) dit++;
            uint64_t median = MADBasedMedian(dit, duration.end());
            if (first) {
                ofs << "\n";
                first = false;
            } else {
                ofs << ",\n";
            }
            ofs << "\t{\"memset\":" << (int)kind
                << ",\"type\":\"" << memset_type_name[kind] << "\""
                << ",\"size\":" << size
                << ",\"median\":" << median
                << ",\"cnt\":" << duration.size()
                << "}";
        }
    }

    ofs << "\n]";
    ofs.close();
}

void PtiRecvKernelRecord(CUpti_ActivityKernel6 *kernel)
{
    XDEBG("PtiRecvKernelRecord: %s", kernel->name);
    int32_t grid_x = kernel->gridX; // max = 2 ^ 31 - 1
    int32_t grid_y = kernel->gridY; // max = 2 ^ 16 - 1
    int32_t grid_z = kernel->gridZ; // max = 2 ^ 16 - 1
    int32_t block_x = kernel->blockX; // max = 1024
    int32_t block_y = kernel->blockY; // max = 1024
    int32_t block_z = kernel->blockZ; // max = 64

    const char *name = kernel->name;
    uint64_t gdim = PackGridDim(grid_x, grid_y, grid_z);
    uint32_t bdim = PackBlockDim(block_x, block_y, block_z);
    int64_t time_ns = kernel->end - kernel->start;

    uint64_t hash = 0;
    auto nit = name_hash.find(name);
    if (nit == name_hash.end()) {
        hash = HashDjb2(name);
        name_hash[name] = hash;
        kernel_name[hash] = std::string(name);
    } else {
        hash = nit->second;
    }

    auto kit = kernel_history.find(hash);
    if (kit == kernel_history.end()) {
        kernel_history[hash][gdim][bdim] = std::list<int64_t>({time_ns});
        return;
    }

    auto git = kit->second.find(gdim);
    if (git == kit->second.end()) {
        kit->second[gdim][bdim] = std::list<int64_t>({time_ns});
        return;
    }

    auto bit = git->second.find(bdim);
    if (bit == git->second.end()) {
        git->second[bdim] = std::list<int64_t>({time_ns});
        if (1 == kProfileCnt) PtiSave();
        return;
    }

    bit->second.push_back(time_ns);
    if (bit->second.size() == kProfileCnt) PtiSave();
}

void PtiRecvMemcpyRecord(CUpti_ActivityMemcpy6 *memcpy)
{
    uint64_t size = memcpy->bytes;
    uint8_t kind = memcpy->copyKind;
    int64_t time_ns = memcpy->end - memcpy->start;

    auto kit = memcpy_history.find(kind);
    if (kit == memcpy_history.end()) {
        memcpy_history[kind][size] = std::list<int64_t>({time_ns});
        return;
    }

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) {
        kit->second[size] = std::list<int64_t>({time_ns});
        if (1 == kProfileCnt) PtiSave();
        return;
    }

    sit->second.push_back(time_ns);
    if (sit->second.size() == kProfileCnt) PtiSave();
}

void PtiRecvMemsetRecord(CUpti_ActivityMemset4 *memset)
{
    uint64_t size = memset->bytes;
    uint16_t kind = memset->memoryKind;
    int64_t time_ns = memset->end - memset->start;

    auto kit = memset_history.find(kind);
    if (kit == memset_history.end()) {
        memset_history[kind][size] = std::list<int64_t>({time_ns});
        return;
    }

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) {
        kit->second[size] = std::list<int64_t>({time_ns});
        if (1 == kProfileCnt) PtiSave();
        return;
    }

    sit->second.push_back(time_ns);
    if (sit->second.size() == kProfileCnt) PtiSave();
}

}
