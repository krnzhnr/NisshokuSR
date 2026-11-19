// Driver-созданный primary-контекст — как в семплах SDK (AppDec/AppEncCuda
// делают cuCtxCreate, а не cudaSetDevice). Runtime API (cudaMalloc, кернелы,
// TRT, streams) подхватывает текущий driver-контекст автоматически.
// Это чинит NV_ENC_ERR_MAP_FAILED на больших кадрах: буферы, зарегистренные
// NVENC под runtime-lazy-primary, на 4K не маппятся, под driver-контекстом — да.
#include "CudaContext.h"

bool CudaContext::init() {
  if (cuInit(0) != CUDA_SUCCESS) return false;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
  // WDDM: BLOCKING_SYNC жрет меньше CPU чем спин.
  if (cuCtxCreate(&primary_ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev) != CUDA_SUCCESS) return false;
  if (cuCtxSetCurrent(primary_ctx) != CUDA_SUCCESS) return false;
  // ВАЖНО: обычные дефолтные стримы. Приоритетный non-blocking стрим
  // роняет TRT-инференс на Blackwell (illegal access на внутренних стримах),
  // что травит весь контекст (NVENC MAP_FAILED, cuvidMap INVALID_*).
  if (cudaStreamCreate(&comp_stream) != cudaSuccess) return false;
  return true;
}

void CudaContext::shutdown() {
  if (primary_ctx) cuCtxSetCurrent(primary_ctx);
  if (comp_stream) { cudaStreamDestroy(comp_stream); comp_stream = nullptr; }
  if (primary_ctx) { cuCtxDestroy(primary_ctx); primary_ctx = nullptr; }
}
