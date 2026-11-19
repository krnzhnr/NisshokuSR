#pragma once
// Primary-контекст + один вычислительный стрим. ОДИН контекст на весь процесс
// (как в семплах SDK): декодер, TRT и энкодеры делят primary, никаких
// cuCtxCreate/pop в обертках — иначе ломается cuvidMapVideoFrame и TRT.
// primaryCtx() оставлен для явных проверок/возвратов при отладке.
#include <cuda_runtime.h>
#include <cuda.h>

struct CudaContext {
  cudaStream_t comp_stream = nullptr;  // pre-kernel + TRT + post-kernel
  CUcontext primary_ctx = nullptr;     // stash после init

  CUcontext primaryCtx() const { return primary_ctx; }
  bool init();
  void shutdown();
  ~CudaContext() { shutdown(); }
};
