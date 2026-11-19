// Обертка над SDK 13.x NvDecoder: штатный парсер, кадры остаются на GPU.
// Требует Samples/NvCodec/NvDecoder/NvDecoder.h в include path.
//
// Контексты: ОДИН primary-контекст на весь процесс (как в семплах SDK).
// NvDecoder получает primary через cuCtxGetCurrent и НЕ владеет им.
// Никаких cuCtxCreate/pop в обертке: лишние driver-контексты ломали
// cuvidMapVideoFrame (INVALID_HANDLE/INVALID_VALUE) и TRT-рантайм.
// Все SDK-вызовы сами push/pop внутри и сбалансированы.
#include "NvDecWrapper.h"
#include "NvDecoder.h"
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

struct NvDecWrapper::Impl {
  NvDecoder* dec = nullptr;
  CUcontext ctx = nullptr; // primary, чужой — не создавать, не разрушать
  CUdevice dev = 0;
  int decode_errors = 0;   // пакеты, которые NVDEC не смог разобрать
};

static cudaVideoCodec CodecOf(const std::string& codec) {
  if (codec == "hevc" || codec == "h265") return cudaVideoCodec_HEVC;
  if (codec == "av1") return cudaVideoCodec_AV1;
  if (codec == "vp9") return cudaVideoCodec_VP9;
  return cudaVideoCodec_H264;
}

bool NvDecWrapper::init(int gpu_id, const std::string& codec) {
  shutdown();
  impl_ = new Impl();
  if (cuInit(0) != CUDA_SUCCESS) { shutdown(); return false; }
  if (cuDeviceGet(&impl_->dev, gpu_id) != CUDA_SUCCESS) { shutdown(); return false; }
  // Primary уже поднят CudaContext::init через Driver API — забираем хендл.
  // cudaSetDevice здесь НЕ вызываем: он переключил бы на runtime-lazy-primary.
  if (cuCtxGetCurrent(&impl_->ctx) != CUDA_SUCCESS || !impl_->ctx) { shutdown(); return false; }
  try {
    impl_->dec = new NvDecoder(impl_->ctx, /*bUseDeviceFrame*/true, CodecOf(codec),
                               /*bLowLatency*/false, /*bDeviceFramePitched*/false,
                               nullptr, nullptr, false,
                               /*maxWidth*/0, /*maxHeight*/0, 1000, false, 0);
  } catch (...) { shutdown(); return false; }
  return true;
}

static void PullReady(NvDecoder* dec, std::vector<DecodedFrame>& out, int n, int64_t pts) {
  for (int i = 0; i < n; ++i) {
    int64_t ts = pts;
    uint8_t* p = dec->GetLockedFrame(&ts);
    if (!p) break;
    DecodedFrame f;
    f.dev_ptr = p; // device NV12
    f.pitch = dec->GetDeviceFramePitch();
    f.w = dec->GetWidth();
    f.h = dec->GetHeight();
    f.bit_depth = dec->GetBitDepth();
    f.pts = ts ? ts : pts;
    out.push_back(f);
  }
}

bool NvDecWrapper::decodePacket(const uint8_t* data, size_t size, int64_t pts,
                                std::vector<DecodedFrame>& out) {
  out.clear();
  if (!impl_ || !impl_->dec || !data || !size) return false;
  // SDK кидает NVDECException (с функцией/файлом/строкой внутри what())
  // на битых пакетах — ронять весь прогон нельзя, скипаем пакет.
  try {
    int n = impl_->dec->Decode(data, (int)size, 0, pts);
    PullReady(impl_->dec, out, n, pts);
  } catch (const NVDECException& e) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] skip packet size=%d: code=%d %s\n", (int)size, (int)e.getErrorCode(), e.what());
  } catch (const std::exception& e) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] skip packet size=%d: %s\n", (int)size, e.what());
  } catch (...) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] skip packet size=%d: unknown\n", (int)size);
  }
  return true;
}

bool NvDecWrapper::flush(std::vector<DecodedFrame>& out) {
  out.clear();
  if (!impl_ || !impl_->dec) return false;
  try {
    int n = impl_->dec->Decode(nullptr, 0); // дрейн парсера
    PullReady(impl_->dec, out, n, 0);
  } catch (const NVDECException& e) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] flush fail: code=%d %s\n", (int)e.getErrorCode(), e.what());
    return false;
  } catch (const std::exception& e) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] flush fail: %s\n", e.what());
    return false;
  } catch (...) {
    ++impl_->decode_errors;
    fprintf(stderr, "[dec] flush fail: unknown\n");
    return false;
  }
  return true;
}

int NvDecWrapper::decodeErrors() const {
  return impl_ ? impl_->decode_errors : 0;
}

void NvDecWrapper::unlock(unsigned char* dev_ptr) {
  if (!impl_ || !impl_->dec || !dev_ptr) return;
  uint8_t* p = dev_ptr;
  impl_->dec->UnlockFrame(&p, 1);
}

void NvDecWrapper::shutdown() {
  if (impl_) {
    delete impl_->dec; // контекст чужой (primary) — не трогаем
    delete impl_;
    impl_ = nullptr;
  }
}
