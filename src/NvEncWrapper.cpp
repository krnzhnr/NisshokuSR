// NvEncoderCuda из SDK 13.x: GetNextInputFrame (device) + EncodeFrame.
// API 13.x: конструктор от CUcontext + NV_ENC_BUFFER_FORMAT, запуск через
// CreateEncoder(), пакеты — NvEncOutputFrame{.frame, .pictureType, .timeStamp}.
// Режим — chunked GOP (см. README): четные GOP на enc0, нечетные на enc1.
// Контексты: ОДИН primary на всех (как в семплах SDK). Все энкодеры делят
// primary-контекст CudaContext — никаких cuCtxCreate/pop в обертке.
// Reorder-буфер: у NVENC есть внутренняя задержка вывода, и хвост чанка
// приезжает уже когда пишет другой энкодер — копим по ГЛОБАЛЬНОМУ pts
// и отдаем строго по порядку (окно depth). Первый кадр чанка —
// принудительный IDR: чанки независимы.
// С B-кадрами пакеты идут в decode-order (POC), переставлять их в display-
// order нельзя: B ссылается на будущий P. Поэтому B-режим всегда односессионный
// и пишется как пришел; chunked dual остается для B=0/lookahead=0.
// ТОЧКА АПГРЕЙДА до AppTransZeroCopy: общий CUarray с
// CUDA_ARRAY3D_VIDEO_ENCODE_DECODE вместо DtoD-копии во входной буфер.
#include "NvEncWrapper.h"
#include "NvEncoderCuda.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <map>
#include <vector>

struct NvEncWrapper::Impl {
  std::vector<NvEncoderCuda*> encs;
  CUcontext ctx = nullptr; // primary, чужой — не создавать, не разрушать
  int w = 0, h = 0;
  int gop = 120;
  int chunk = 120;      // кадров на один энкодер до переключения
  int active = 0;       // какой энкодер сейчас пишет
  int in_chunk = 0;     // сколько кадров уже отдано активному
  int bframes = 0;
  int bit_depth = 8;
  // Reorder-буфер: задержка вывода NVENC (до ~4 кадров при B=0 ULL).
  std::map<int64_t, std::vector<EncPacket>> backlog; // глобальный pts -> пакеты
  int64_t max_in_pts = -1;
  int64_t next_out_pts = 0;
  bool started = false;
};

static GUID PresetGuid(const std::string& preset) {
  if (preset == "p7" || preset == "quality") return NV_ENC_PRESET_P7_GUID;
  if (preset == "p6") return NV_ENC_PRESET_P6_GUID;
  if (preset == "p5") return NV_ENC_PRESET_P5_GUID;
  if (preset == "p4") return NV_ENC_PRESET_P4_GUID;
  if (preset == "p3") return NV_ENC_PRESET_P3_GUID;
  if (preset == "p2") return NV_ENC_PRESET_P2_GUID;
  return NV_ENC_PRESET_P1_GUID; // "p1"/"performance"
}

static GUID CodecGuid(const std::string& codec) {
  if (codec == "hevc" || codec == "h265") return NV_ENC_CODEC_HEVC_GUID;
  if (codec == "av1") return NV_ENC_CODEC_AV1_GUID;
  return NV_ENC_CODEC_H264_GUID;
}

// NVEncC --split-enc: разбиение кадра между NVENC-движками.
static NV_ENC_SPLIT_ENCODE_MODE SplitMode(const std::string& mode) {
  if (mode == "auto") return NV_ENC_SPLIT_AUTO_MODE;
  if (mode == "auto_forced") return NV_ENC_SPLIT_AUTO_FORCED_MODE;
  if (mode == "forced_2" || mode == "two") return NV_ENC_SPLIT_TWO_FORCED_MODE;
  if (mode == "forced_3" || mode == "three") return NV_ENC_SPLIT_THREE_FORCED_MODE;
  return NV_ENC_SPLIT_DISABLE_MODE; // "disable"/"off"/неизвестное
}

// Выгрести из backlog подряд идущие pts начиная с next_out_pts.
// Дырка означает потерянный кадр — для файла это дефект, а не повод
// молча писать дальше: возвращаем false, Pipeline пометит файл failed.
bool NvEncWrapper::drainContiguous(std::vector<EncPacket>& out) {
  while (true) {
    auto it = impl_->backlog.find(impl_->next_out_pts);
    if (it != impl_->backlog.end()) {
      for (auto& e : it->second) out.push_back(std::move(e));
      impl_->backlog.erase(it);
      ++impl_->next_out_pts;
      continue;
    }
    if (impl_->max_in_pts - impl_->next_out_pts > 64) {
      fprintf(stderr, "[enc] нет пакета для pts=%lld: кадр потерян, файл прерван\n",
              (long long)impl_->next_out_pts);
      return false;
    }
    break;
  }
  return true;
}

bool NvEncWrapper::init(int gpu_id, int w, int h, int fps_num, int fps_den,
                        const EncoderConfig& cfg) {
  shutdown();
  int num_encoders = cfg.num_encoders;
  if (num_encoders < 1) num_encoders = 1;
  if (num_encoders > 2) num_encoders = 2; // 5070 Ti: ровно 2x Gen9
  // Chunked-режим требует немедленного вывода: хвост чанка должен приехать
  // до переключения на другой энкодер. B-кадры/lookahead держат кадры внутри
  // сессии, поэтому такие настройки — только одна сессия.
  if (num_encoders > 1 && (cfg.bframes > 0 || cfg.lookahead > 0)) {
    fprintf(stderr, "[enc] B-frames/lookahead require single NVENC session; num_encoders %d -> 1\n",
            num_encoders);
    num_encoders = 1;
  }
  if (cuInit(0) != CUDA_SUCCESS) return false;
  impl_ = new Impl();
  impl_->w = w; impl_->h = h;
  impl_->gop = cfg.gop > 0 ? cfg.gop : 120;
  impl_->chunk = impl_->gop; // чанк = GOP: каждый чанк начинается с IDR
  impl_->bframes = std::clamp(cfg.bframes, 0, 5); // NVENC: до 5 B-кадров
  if (cfg.bframes > impl_->bframes)
    fprintf(stderr, "[enc] --bframes %d capped to %d (NVENC max)\n", cfg.bframes, impl_->bframes);
  // Primary уже поднят CudaContext::init через Driver API — все сессии делят его.
  // cudaSetDevice здесь НЕ вызываем: он переключил бы на runtime-lazy-primary.
  if (cuCtxGetCurrent(&impl_->ctx) != CUDA_SUCCESS || !impl_->ctx) { shutdown(); return false; }
  GUID guid = CodecGuid(cfg.codec);
  GUID presetGuid = PresetGuid(cfg.preset);
  NV_ENC_TUNING_INFO tuneGuid = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
  if (cfg.tuning == "hq" || cfg.tuning == "quality") tuneGuid = NV_ENC_TUNING_INFO_HIGH_QUALITY;
  else if (cfg.tuning == "ll") tuneGuid = NV_ENC_TUNING_INFO_LOW_LATENCY;
  int bitDepth = cfg.bit_depth >= 10 ? 10 : 8;
  if (bitDepth == 10 && guid == NV_ENC_CODEC_H264_GUID) {
    fprintf(stderr, "[enc] 10-bit не поддерживается для H.264, выключаю\n");
    bitDepth = 8;
  }
  impl_->bit_depth = bitDepth;
  NV_ENC_BUFFER_FORMAT bufFmt = bitDepth == 10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT
                                                : NV_ENC_BUFFER_FORMAT_NV12;
  NV_ENC_SPLIT_ENCODE_MODE splitMode = SplitMode(cfg.split_enc);
  if (splitMode != NV_ENC_SPLIT_DISABLE_MODE && guid == NV_ENC_CODEC_H264_GUID) {
    fprintf(stderr, "[enc] split-enc не поддерживается для H.264, выключаю\n");
    splitMode = NV_ENC_SPLIT_DISABLE_MODE;
  }
  // Одна сессия уже использует два движка; chunked dual поверх этого не нужен.
  if (splitMode != NV_ENC_SPLIT_DISABLE_MODE && num_encoders > 1) {
    fprintf(stderr, "[enc] split-enc использует оба NVENC; num_encoders %d -> 1\n", num_encoders);
    num_encoders = 1;
  }
  int refs = std::clamp(cfg.refs, 0, 16); // DPB; NVEncC --ref 8 = maxNumRefFramesInDPB
  if (cfg.refs > refs)
    fprintf(stderr, "[enc] --ref %d capped to %d (DPB max)\n", cfg.refs, refs);
  // SVFI: --lookahead 16 (Adaptive I + B Insert включаются драйвером).
  // Верхняя граница NVENC: 31 - число B-кадров.
  int lookahead = std::clamp(cfg.lookahead, 0, 31 - impl_->bframes);
  if (cfg.lookahead > lookahead)
    fprintf(stderr, "[enc] --lookahead %d capped to %d (31 - B frames)\n", cfg.lookahead, lookahead);
  try {
    for (int i = 0; i < num_encoders; ++i) {
      // nExtraOutputDelay: при B-кадрах нужен запас >=1 (SDK-дефолт 3),
      // чтобы энкодер и графика работали параллельно; без B — 0, как у
      // AppEncLowLatency (синхронный вывод, хвост чанка не запаздывает).
      uint32_t extraDelay = impl_->bframes > 0 ? 3 : 0;
      NvEncoderCuda* e = new NvEncoderCuda(impl_->ctx, w, h, bufFmt, extraDelay);
      NV_ENC_INITIALIZE_PARAMS ip = {};
      NV_ENC_CONFIG cfgNv = {};
      ip.encodeConfig = &cfgNv;
      e->CreateDefaultEncoderParams(&ip, guid, presetGuid, tuneGuid);
      if (cfg.rc == "cqp" || cfg.rc == "constqp") {
        // Качество-приоритет как у SVFI (у них CQP 18 + quality preset).
        ip.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        uint32_t q = (uint32_t)(cfg.qp > 0 ? cfg.qp : 18);
        ip.encodeConfig->rcParams.constQP = {q, q, q};
      } else {
        // CBR: битрейт держит драйвер; max/vbv не трогаем (как у эталона).
        ip.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        ip.encodeConfig->rcParams.averageBitRate = (uint32_t)cfg.bitrate_kbps * 1000u;
      }
      ip.frameRateNum = (uint32_t)fps_num; ip.frameRateDen = (uint32_t)fps_den;
      ip.encodeConfig->gopLength = (uint32_t)impl_->gop;
      ip.encodeConfig->frameIntervalP = impl_->bframes + 1;
      ip.splitEncodeMode = splitMode;
      ip.encodeConfig->rcParams.enableLookahead = lookahead > 0 ? 1 : 0;
      ip.encodeConfig->rcParams.lookaheadDepth = (uint16_t)lookahead;
      ip.encodeConfig->rcParams.lookaheadLevel = NV_ENC_LOOKAHEAD_LEVEL_0;
      // Split encoding несовместим с weighted prediction (шапка SDK).
      if (splitMode != NV_ENC_SPLIT_DISABLE_MODE) ip.enableWeightedPrediction = 0;
      // B-кадры несовместимы с weighted prediction (NVENC header).
      if (impl_->bframes > 0) ip.enableWeightedPrediction = 0;
      // VUI как у SVFI: --colorrange tv --colorprim/--colormatrix/--transfer bt709.
      // Плюс timing info: без него raw-демуксер FFmpeg не восстанавливает PTS
      // у B-кадров (unknown timestamp при ремуксе).
      auto setVui = [&](NV_ENC_CONFIG_H264_VUI_PARAMETERS& vui, bool hevc) {
        vui.videoSignalTypePresentFlag = 1;
        vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        vui.videoFullRangeFlag = 0; // limited/tv
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        vui.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;
        vui.timingInfoPresentFlag = 1;
        vui.numUnitInTicks = (uint32_t)fps_den;
        vui.timeScale = (uint32_t)(hevc ? fps_num : fps_num * 2);
      };
      if (guid == NV_ENC_CODEC_HEVC_GUID) {
        auto& hevc = ip.encodeConfig->encodeCodecConfig.hevcConfig;
        hevc.tier = 0; // main, как --tier main у NVEncC
        hevc.maxNumRefFramesInDPB = (uint32_t)refs; // NVEncC --ref N
        hevc.chromaFormatIDC = 1; // 4:2:0
        hevc.inputBitDepth = bitDepth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
        hevc.outputBitDepth = bitDepth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
        if (bitDepth == 10) ip.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        setVui(hevc.hevcVUIParameters, true);
        if (impl_->bframes > 0) {
          hevc.useBFramesAsRef = cfg.bref_middle ? NV_ENC_BFRAME_REF_MODE_MIDDLE
                                                 : NV_ENC_BFRAME_REF_MODE_DISABLED;
        }
        fprintf(stderr, "[enc] hevc refs dpb=%u l0=%d l1=%d depth=%d\n",
                hevc.maxNumRefFramesInDPB, (int)hevc.numRefL0, (int)hevc.numRefL1, bitDepth);
      } else if (guid == NV_ENC_CODEC_H264_GUID) {
        auto& h264 = ip.encodeConfig->encodeCodecConfig.h264Config;
        h264.maxNumRefFrames = (uint32_t)refs; // NVEncC --ref N
        setVui(h264.h264VUIParameters, false);
        if (impl_->bframes > 0) {
          h264.useBFramesAsRef = cfg.bref_middle ? NV_ENC_BFRAME_REF_MODE_MIDDLE
                                                 : NV_ENC_BFRAME_REF_MODE_DISABLED;
        }
      } else if (guid == NV_ENC_CODEC_AV1_GUID) {
        auto& av1 = ip.encodeConfig->encodeCodecConfig.av1Config;
        av1.chromaFormatIDC = 1;
        av1.inputBitDepth = bitDepth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
        av1.outputBitDepth = bitDepth == 10 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
      }
      fprintf(stderr, "[enc] cfg %dx%d fps=%d/%d gop=%d bf=%d ref=%d la=%d bref=%d split=%d depth=%d "
                      "preset=%s tune=%d rc=%d qp=%u avg=%u mp=%d aq=%d taaq=%d sessions=%d\n",
          w, h, fps_num, fps_den, impl_->gop, impl_->bframes, refs, lookahead,
          (int)cfg.bref_middle, (int)splitMode, bitDepth, cfg.preset.c_str(), (int)tuneGuid,
          (int)ip.encodeConfig->rcParams.rateControlMode,
          (unsigned)ip.encodeConfig->rcParams.constQP.qpInterP,
          ip.encodeConfig->rcParams.averageBitRate,
          (int)ip.encodeConfig->rcParams.multiPass,
          (int)ip.encodeConfig->rcParams.enableAQ,
          (int)ip.encodeConfig->rcParams.enableTemporalAQ, num_encoders);
      e->CreateEncoder(&ip);
      impl_->encs.push_back(e);
    }
  } catch (...) { shutdown(); return false; }
  return true;
}

int NvEncWrapper::numEncoders() const { return impl_ ? (int)impl_->encs.size() : 0; }

bool NvEncWrapper::encodeFromDevice(unsigned char* nv12_dev, int pitch,
                                     int64_t pts, std::vector<EncPacket>& out) {
  out.clear();
  if (!impl_ || impl_->encs.empty()) return false;
  if (!impl_->started) { impl_->next_out_pts = pts; impl_->started = true; }
  // Переключение чанка: следующий кадр начнет новый GOP -> IDR у другого энкодера.
  if (impl_->encs.size() > 1 && impl_->in_chunk >= impl_->chunk) {
    impl_->active = (impl_->active + 1) % (int)impl_->encs.size();
    impl_->in_chunk = 0;
  }
  NvEncoderCuda* enc = impl_->encs[impl_->active];
  const NvEncInputFrame* in = enc->GetNextInputFrame();
  // Ручная 3-плоскостная DtoD-копия (luma + interleaved chroma) вместо
  // SDK CopyToDeviceFrame: тот же результат, без лишнего push/pop.
  // DtoH тут нет намеренно: конвейеру чтение на хост не нужно вообще.
  // 10-bit: P010, 2 байта на сэмпл; ширину в байтах удваиваем.
  size_t sampleBytes = impl_->bit_depth == 10 ? 2 : 1;
  // Копия идемпотентна: при сбое её безопасно повторить (кадр в NVENC
  // ещё не ушёл). Повторяем один раз, дальше — ошибка файла.
  cudaError_t c1 = cudaSuccess, c2 = cudaSuccess;
  for (int attempt = 0; attempt < 2; ++attempt) {
    c1 = cudaMemcpy2D(in->inputPtr, in->pitch,
        nv12_dev, (size_t)pitch, (size_t)impl_->w * sampleBytes, (size_t)impl_->h, cudaMemcpyDeviceToDevice);
    c2 = cudaMemcpy2D((uint8_t*)in->inputPtr + in->chromaOffsets[0], in->pitch,
        nv12_dev + (size_t)pitch * (size_t)impl_->h, (size_t)pitch,
        (size_t)impl_->w * sampleBytes, (size_t)impl_->h / 2, cudaMemcpyDeviceToDevice);
    if (c1 == cudaSuccess && c2 == cudaSuccess) break;
    if (attempt == 0) {
      fprintf(stderr, "[enc] copy fail luma=%d chroma=%d buf=%d, retry\n",
              (int)c1, (int)c2, impl_->active);
      cudaGetLastError(); // сбросить last-error перед повтором
    }
  }
  if (c1 != cudaSuccess || c2 != cudaSuccess) {
    fprintf(stderr, "[enc] copy fail luma=%d chroma=%d buf=%d: кадр не закодирован, файл прерван\n",
            (int)c1, (int)c2, impl_->active);
    return false;
  }
  std::vector<NvEncOutputFrame> vpkt;
  // SDK кидает NVENCException. Для файла это не «пропустить кадр», а ошибка:
  // дырка по PTS хуже явного failed, поэтому вызывающий прервёт файл.
  // Первый кадр чанка — принудительный IDR: чанки независимы даже если
  // внутренние GOP-счетчики энкодеров разъехались.
  NV_ENC_PIC_PARAMS picParams = {};
  picParams.version = NV_ENC_PIC_PARAMS_VER;
  // Непрозрачный timestamp: NVENC вернет его в outputTimeStamp, по нему
  // восстанавливаем display-pts при B-кадрах. 0 зарезервирован (автоинкремент).
  picParams.inputTimeStamp = (uint64_t)(pts + 1);
  if (impl_->encs.size() > 1 && impl_->in_chunk == 0) picParams.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
  try {
    enc->EncodeFrame(vpkt, &picParams);
  } catch (const NVENCException& e) {
    fprintf(stderr, "[enc] encode fail pts=%lld: code=%d %s; файл прерван\n",
            (long long)pts, (int)e.getErrorCode(), e.what());
    return false;
  } catch (const std::exception& e) {
    fprintf(stderr, "[enc] encode fail pts=%lld: %s; файл прерван\n",
            (long long)pts, e.what());
    return false;
  } catch (...) {
    fprintf(stderr, "[enc] encode fail pts=%lld: unknown; файл прерван\n",
            (long long)pts);
    return false;
  }
  impl_->in_chunk++;
  for (auto& o : vpkt) {
    EncPacket e;
    e.data = std::move(o.frame);
    e.pts = o.timeStamp ? (int64_t)o.timeStamp - 1 : pts;
    e.key = (o.pictureType == NV_ENC_PIC_TYPE_IDR);
    if (impl_->bframes > 0) {
      // Decode-order: B ссылается на будущий P, перестановка сломает поток.
      out.push_back(std::move(e));
    } else {
      impl_->backlog[e.pts].push_back(std::move(e));
    }
  }
  if (pts > impl_->max_in_pts) impl_->max_in_pts = pts;
  if (impl_->bframes == 0 && !drainContiguous(out)) return false;
  return true;
}

bool NvEncWrapper::flush(std::vector<EncPacket>& out) {
  out.clear();
  if (!impl_) return false;
  bool ok = true;
  for (auto* enc : impl_->encs) {
    std::vector<NvEncOutputFrame> vpkt;
    try {
      enc->EndEncode(vpkt);
    } catch (const std::exception& e) {
      fprintf(stderr, "[enc] flush fail: %s\n", e.what());
      ok = false;
      continue;
    } catch (...) {
      fprintf(stderr, "[enc] flush fail: unknown\n");
      ok = false;
      continue;
    }
    for (auto& o : vpkt) {
      EncPacket e;
      e.data = std::move(o.frame);
      e.pts = o.timeStamp ? (int64_t)o.timeStamp - 1 : ++impl_->max_in_pts;
      e.key = (o.pictureType == NV_ENC_PIC_TYPE_IDR);
      if (impl_->bframes > 0) out.push_back(std::move(e));
      else impl_->backlog[e.pts].push_back(std::move(e));
    }
  }
  if (impl_->bframes == 0) {
    // Хвост: отдаем остаток строго по возрастанию pts.
    for (auto& kv : impl_->backlog) {
      for (auto& e : kv.second) out.push_back(std::move(e));
    }
    impl_->backlog.clear();
  }
  return ok;
}

void NvEncWrapper::shutdown() {
  if (impl_) {
    for (auto* e : impl_->encs) delete e;
    impl_->encs.clear();
    // Контекст чужой (primary CudaContext) — не разрушаем.
    delete impl_;
    impl_ = nullptr;
  }
}
