#include <cstdint>
#include <fstream>
#include <json/json.h>

#include "xsched/utils/xassert.h"
#include "xsched/preempt/xmem/xmem.h"
#include "xsched/cuda/hal/common/dims.h"
#include "xsched/cuda/hal/common/kernel.h"
#include "xsched/cuda/hal/profile/profile.h"
#include "xsched/cuda/hal/profile/relation.h"
#include "xsched/cuda/hal/profile/protocol.h"

using namespace xsched::cuda;
using namespace xsched::utils;
using namespace xsched::preempt;

Profiler Profiler::g_profiler;

Profiler::Profiler()
{
    Json::Reader reader;

    // load profiled latencies
    std::ifstream latency_file(GetExecutionTimeFileName());
    Json::Value latencies;
    reader.parse(latency_file, latencies);
    latency_file.close();

    for (auto &obj : latencies) {
        if (obj.isMember("kernel")) {
            uint64_t hash = obj["kernel"].asUInt64();
            auto &gdim = obj["gdim"];
            auto &bdim = obj["bdim"];
            XASSERT(gdim.size() == 3, "gdim size should be 3");
            XASSERT(bdim.size() == 3, "bdim size should be 3");
            uint64_t gdim_pack = PackGridDim(gdim[0].asInt(), gdim[1].asInt(), gdim[2].asInt());
            uint32_t bdim_pack = PackBlockDim(bdim[0].asInt(), bdim[1].asInt(), bdim[2].asInt());
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

    // load profiled memory access
    std::ifstream memory_access_file(GetMemoryContextHeuristicFileName());
    Json::Value memory_accesses;
    reader.parse(memory_access_file, memory_accesses);
    memory_access_file.close();

    for (auto it = memory_accesses.begin(); it != memory_accesses.end(); it++) {
        std::string key = it.name();
        uint64_t hash = HashDjb2(key.c_str());
        Json::Value &relations = *it;
        XASSERT(relations.isArray(), "relations should be an array");
        for (auto rit = relations.begin(); rit != relations.end(); rit++) {
            Json::Value &relation = *rit;
            auto &gdim = relation["gdim"];
            auto &bdim = relation["bdim"];
            XASSERT(gdim.isArray() && gdim.size() == 3, "gdim size should be 3");
            XASSERT(bdim.isArray() && bdim.size() == 3, "bdim size should be 3");
            uint64_t gdim_pack = PackGridDim(gdim[0].asInt(), gdim[1].asInt(), gdim[2].asInt());
            uint32_t bdim_pack = PackBlockDim(bdim[0].asInt(), bdim[1].asInt(), bdim[2].asInt());
            
            XASSERT(relation.isMember("read") && relation["read"].isArray(),
                    "read should be an array");
            XASSERT(relation.isMember("write") && relation["write"].isArray(),
                    "write should be an array");
            Json::Value &reads = relation["read"];
            Json::Value &writes = relation["write"];
            for (auto read_it = reads.begin(); read_it != reads.end(); read_it++) {
                relations_[hash][gdim_pack][bdim_pack].first.push_back(
                    AccessRelation::Create(*read_it));
            }
            for (auto write_it = writes.begin(); write_it != writes.end(); write_it++) {
                relations_[hash][gdim_pack][bdim_pack].second.push_back(
                    AccessRelation::Create(*write_it));
            }
        }
    }
}

Profiler::~Profiler()
{

}

bool Profiler::GetKernelMemAccessInternal(
    utils::Region &reads, utils::Region &writes, CUcontext ctx,
    CUfunction func,      uint64_t pgdim,      uint32_t pbdim,      void **params,
    CUfunction func_prev, uint64_t pgdim_prev, uint32_t pbdim_prev, void **params_prev) const
{
    uint64_t hash = GetKernelNameHash(ctx, func);
    uint64_t hash_prev = GetKernelNameHash(ctx, func_prev);

    auto kit = relations_.find(hash);
    if (kit == relations_.end()) return false;
    auto git = kit->second.find(pgdim);
    if (git == kit->second.end()) return false;
    auto bit = git->second.find(pbdim);
    if (bit == git->second.end()) return false;
    auto &rw_relations = bit->second;

    for (auto &relation : rw_relations.first) {
        relation->MergeRegion(reads, hash_prev, pgdim_prev, pbdim_prev, params_prev, params);
    }
    for (auto &relation : rw_relations.second) {
        relation->MergeRegion(writes, hash_prev, pgdim_prev, pbdim_prev, params_prev, params);
    }

    return true;
}

int64_t Profiler::GetKernelExecutionTimeNsInternal(CUcontext ctx, CUfunction func,
                                                   int32_t gdx, int32_t gdy, int32_t gdz,
                                                   int32_t bdx, int32_t bdy, int32_t bdz) const
{
    uint64_t hash = GetKernelNameHash(ctx, func);
    if (hash == 0) return -1;

    auto kit = kernel_.find(hash);
    if (kit == kernel_.end()) return -1;

    uint64_t gdim = PackGridDim(gdx, gdy, gdz);
    auto git = kit->second.find(gdim);
    if (git == kit->second.end()) return -1;

    uint32_t bdim = PackBlockDim(bdx, bdy, bdz);
    auto bit = git->second.find(bdim);
    if (bit == git->second.end()) return -1;
    return bit->second;
}

int64_t Profiler::GetMemcpyExecutionTimeNsInternal(CUpti_ActivityMemcpyKind kind,
                                                   size_t size) const
{
    auto kit = memcpy_.find(kind);
    if (kit == memcpy_.end()) return -1;

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) return -1;
    return sit->second;
}

int64_t Profiler::GetMemsetExecutionTimeNsInternal(CUpti_ActivityMemoryKind kind,
                                                   size_t size) const
{
    auto kit = memset_.find(kind);
    if (kit == memset_.end()) return -1;

    auto sit = kit->second.find(size);
    if (sit == kit->second.end()) return -1;
    return sit->second;
}
