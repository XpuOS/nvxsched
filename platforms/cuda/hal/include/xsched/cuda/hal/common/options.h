#pragma once

namespace xsched::cuda
{

enum CudaLv3Implementation
{
    kCudaLv3ImplementationTrap,
    kCudaLv3ImplementationTsg,
};

bool GetCudaUvmEnabled();
bool GetCudaPtdsAutoXQueueEnabled();
bool GetCudaSingleStreamPerProcessEnabled();
bool GetCudaSinglePtdsPerProcessEnabled();
CudaLv3Implementation GetCudaLv3Implementation();

} // namespace xsched::cuda
