// Рабочий вариант: один управляющий тред, все на GPU.
// Decode (NVDEC) -> pre-kernel -> TRT enqueueV3 -> post-kernel -> NVENC.
// Хоста в цикле нет, только DtoD. Трёхтредовый разнос не реализован:
// очередь кадров и пул буферов под него в коде отсутствуют.
#include "Pipeline.h"
#include "CudaContext.h"
#include "Demuxer.h"
#include "Muxer.h"
#include "NvDecWrapper.h"
#include "NvEncWrapper.h"
#include "TrtUpscaler.h"
#include "WinThread.h"
#include "pre_post.h"
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <windows.h>

static bool HasSuffixNoCase(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  if (s.size() < n) return false;
  for (size_t i = 0; i < n; ++i) {
    unsigned char a = (unsigned char)s[s.size() - n + i];
    unsigned char b = (unsigned char)suffix[i];
    if (std::tolower(a) != std::tolower(b)) return false;
  }
  return true;
}

static bool IsMuxedPath(const std::string& p) {
  return HasSuffixNoCase(p, ".mkv") || HasSuffixNoCase(p, ".mp4");
}

// Raw-выход открываем wide-путём: cfg.out_path — UTF-8, а CRT fopen
// на Windows трактует narrow-путь в ACP (кириллица ломается).
static FILE* OpenUtf8(const std::string& path, const char* mode) {
  const auto to_wide = [](const char* s) -> std::wstring {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    w.resize((size_t)n - 1);
    return w;
  };
  const std::wstring wp = to_wide(path.c_str());
  const std::wstring wm = to_wide(mode);
  if (wp.empty() || wm.empty()) return nullptr;
  return _wfopen(wp.c_str(), wm.c_str());
}

static bool WriteAll(FILE* f, const void* data, size_t size) {
  return size == 0 || std::fwrite(data, 1, size, f) == size;
}

bool Pipeline::run(const PipelineConfig& cfg) {
  PinCurrentThread(2, THREAD_PRIORITY_HIGHEST);
  EnableMmcssPlayback();

  Demuxer demux;
  int W = 0, H = 0;
  std::string cc;
  double fps = 30;
  int fps_n = 0, fps_d = 0;
  if (!demux.open(cfg.in_path, &W, &H, &cc, &fps, &fps_n, &fps_d)) {
    printf("demux open fail\n");
    return false;
  }
  if (fps_n <= 0 || fps_d <= 0) { fps_n = 30; fps_d = 1; }
  printf("in %dx%d codec=%s fps=%.3f (%d/%d)\n", W, H, cc.c_str(), fps, fps_n, fps_d);

  CudaContext cu;
  if (!cu.init()) { printf("cuda init fail\n"); return false; }

  // Порядок важен: TRT инициализируем пока текущий — primary (до создания
  // driver-контекстов декодера/энкодеров), иначе рантайм привяжется не туда.
  TrtUpscaler trt;
  TrtUpscaleOptions trt_opt;
  trt_opt.engine_path = cfg.engine_path;
  trt_opt.onnx_path = cfg.onnx_path;
  trt_opt.in_w = W;
  trt_opt.in_h = H;
  trt_opt.scale = cfg.scale;
  trt_opt.rebuild = cfg.rebuild_engine;
  std::string trt_error;
  if (!trt.init(trt_opt, trt_error)) {
    fprintf(stderr, "[engine] %s\n", trt_error.c_str());
    return false;
  }
  int oW = trt.outW(), oH = trt.outH();
  printf("upscale %dx%d -> %dx%d\n", W, H, oW, oH);
  if (W > 0 && cfg.scale > 0 &&
      std::abs((double)oW / (double)W - (double)cfg.scale) > 1e-6) {
    printf("[engine] фактический масштаб %.3fx (--scale %d не применяется)\n",
           (double)oW / (double)W, cfg.scale);
  }

  NvDecWrapper dec;
  if (!dec.init(0, cc)) { printf("nvdec init fail\n"); return false; }

  int bit_depth = cfg.bit_depth >= 10 ? 10 : 8;
  if (bit_depth == 10 && cfg.out_codec == "h264") {
    fprintf(stderr, "[cfg] 10-bit не поддерживается для H.264, ставлю 8-bit\n");
    bit_depth = 8;
  }
  EncoderConfig ecfg;
  ecfg.codec = cfg.out_codec;
  ecfg.preset = cfg.preset;
  ecfg.tuning = cfg.tune;
  ecfg.rc = cfg.rc;
  ecfg.bitrate_kbps = cfg.bitrate_kbps;
  ecfg.gop = cfg.gop;
  ecfg.qp = cfg.qp;
  ecfg.bframes = cfg.bframes;
  ecfg.refs = cfg.refs;
  ecfg.lookahead = cfg.lookahead;
  ecfg.bref_middle = cfg.bref_middle;
  ecfg.split_enc = cfg.split_enc;
  ecfg.bit_depth = bit_depth;
  ecfg.num_encoders = cfg.num_encoders;
  NvEncWrapper enc;
  if (!enc.init(0, oW, oH, fps_n, fps_d, ecfg)) {
    printf("nvenc init fail\n"); return false;
  }
  printf("nvenc %s %dx%d %d-bit, sessions=%d\n",
         cfg.out_codec.c_str(), oW, oH, bit_depth, enc.numEncoders());
  // Контекст один (primary) на всех — декодер, TRT, энкодеры — и текущий
  // не меняется никем (SDK-вызовы push/pop внутри себя сбалансированы).

  // TRT I/O один раз, reuse весь прогон. Аллокация через Driver API,
  // чтобы память точно жила в том же контексте, что и SDK-буферы.
  void *trt_in = nullptr, *trt_out = nullptr, *enc_nv12 = nullptr;
  size_t enc_pitch = bit_depth == 10 ? (((size_t)oW * 2 + 31) & ~(size_t)31)
                                     : (((size_t)oW + 31) & ~(size_t)31);
  size_t enc_bytes = enc_pitch * (size_t)oH * 3 / 2;
  auto free_dev = [&]() {
    if (trt_in) { cuMemFree((CUdeviceptr)trt_in); trt_in = nullptr; }
    if (trt_out) { cuMemFree((CUdeviceptr)trt_out); trt_out = nullptr; }
    if (enc_nv12) { cuMemFree((CUdeviceptr)enc_nv12); enc_nv12 = nullptr; }
  };
  auto alloc_dev = [&](size_t bytes, void** dst, const char* name) -> bool {
    CUdeviceptr p = 0;
    const CUresult r = cuMemAlloc(&p, bytes);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "[cuda] cuMemAlloc %s (%zu MB) failed: %d\n", name,
              bytes >> 20, (int)r);
      return false;
    }
    *dst = (void*)p;
    return true;
  };
  if (!alloc_dev(trt.inBytes(), &trt_in, "trt_in") ||
      !alloc_dev(trt.outBytes(), &trt_out, "trt_out") ||
      !alloc_dev(enc_bytes, &enc_nv12, "enc_nv12")) {
    free_dev();
    return false;
  }
  // CUTENSOR-ядро читает за концом входного тензора (README, грабли №2).
  // Сам тензор целиком пишет pre-kernel, поэтому на кадр зануляем только
  // slack-хвост, а не весь буфер: экономим ~60 МБ трафика на кадр 1080p->4K.
  const size_t trt_in_data = trt.inDataBytes();
  const size_t trt_out_data = trt.outDataBytes();
  const size_t trt_in_slack = trt.inBytes() - trt_in_data;
  const size_t trt_out_slack = trt.outBytes() - trt_out_data;

  const bool muxed = IsMuxedPath(cfg.out_path);
  Muxer muxer;
  FILE* fout = nullptr;
  if (muxed) {
    MuxConfig mc;
    mc.out_path = cfg.out_path;
    mc.source_path = cfg.in_path;
    mc.codec = cfg.out_codec;
    mc.width = oW;
    mc.height = oH;
    mc.fps_num = fps_n;
    mc.fps_den = fps_d;
    mc.bit_depth = bit_depth;
    mc.copy_audio = cfg.copy_audio;
    mc.copy_subs = cfg.copy_subs;
    mc.trim_to_video = cfg.max_frames > 0;
    mc.source_start_us = demux.startTimeUs();
    mc.source_video_start_us = demux.videoStartTimeUs();
    std::string err;
    if (!muxer.open(mc, err)) {
      printf("mux open fail: %s\n", err.c_str());
      free_dev();
      return false;
    }
  } else {
    fout = OpenUtf8(cfg.out_path, "wb");
    if (!fout) {
      printf("open out fail: %s\n", cfg.out_path.c_str());
      free_dev();
      return false;
    }
  }

  Packet pkt;
  int64_t frame_no = 0;
  std::vector<DecodedFrame> frames;
  std::vector<EncPacket> epkts;
  // Один кадр сквозь compute+encode. Возвращает false только на фаталке.
  auto processFrame = [&](DecodedFrame& f) -> bool {
    // NVDEC-кадр разблокируется ровно один раз: явно после DtoD-копии
    // в энкодер либо RAII на любом return false/исключении.
    struct UnlockOnExit {
      NvDecWrapper& dec;
      unsigned char* ptr;
      bool armed = true;
      ~UnlockOnExit() { if (armed && ptr) dec.unlock(ptr); }
      void release() { armed = false; }
    } unlock{dec, f.dev_ptr};

    // Все живут на одном primary-контексте — переключать нечего.
    if (cfg.gray) {
      // Диагностика: без compute и TRT — серый кадр. Проверяет связку
      // decode -> (fill) -> encode без нейросети и кернелов.
      if (bit_depth == 10) {
        launch_fill_p010((unsigned char*)enc_nv12, enc_bytes, 0x8000, cu.comp_stream);
        if (cudaStreamSynchronize(cu.comp_stream) != cudaSuccess) return false;
      } else if (cudaMemset(enc_nv12, 0x80, enc_bytes) != cudaSuccess) {
        return false;
      }
    } else {
      // Зануляем только slack: сам входной тензор полностью перезаписывает
      // pre-kernel. Если CUTENSOR прочитает NaN-паттерн из хвоста, он
      // отравит весь выход (inf/nan) — поэтому хвост чистый каждый кадр.
      if (trt_in_slack &&
          cudaMemsetAsync((char*)trt_in + trt_in_data, 0, trt_in_slack,
                          cu.comp_stream) != cudaSuccess)
        return false;
      if (f.bit_depth >= 10)
        launch_p016_to_rgb_planar_fp16(f.dev_ptr, f.pitch, (__half*)trt_in, W, H, cu.comp_stream);
      else
        launch_nv12_to_rgb_planar_fp16(f.dev_ptr, f.pitch, (__half*)trt_in, W, H, cu.comp_stream);
      cudaError_t ke = cudaGetLastError();
      if (ke != cudaSuccess) {
        printf("pre kernel fail %d\n", (int)ke);
        return false;
      }
      // Выходной тензор пишет engine; зануляем только запас, если ядро
      // читает за его концом. Сам тензор каждый кадр не трогаем.
      if (trt_out_slack &&
          cudaMemsetAsync((char*)trt_out + trt_out_data, 0, trt_out_slack,
                          cu.comp_stream) != cudaSuccess)
        return false;
      if (!trt.inferAsync(trt_in, trt_out, cu.comp_stream)) {
        printf("infer fail\n");
        return false;
      }
    }
    if (!cfg.gray) {
      if (bit_depth == 10)
        launch_rgb_planar_fp16_to_p010((__half*)trt_out, (unsigned char*)enc_nv12, (int)enc_pitch, oW, oH, cu.comp_stream);
      else
        launch_rgb_planar_fp16_to_nv12((__half*)trt_out, (unsigned char*)enc_nv12, (int)enc_pitch, oW, oH, cu.comp_stream);
      cudaError_t se = cudaStreamSynchronize(cu.comp_stream); // точка роста: заменить на events + отдельный encode-тред
      if (se != cudaSuccess) {
        printf("compute stream fail %d\n", (int)se);
        return false;
      }
    }
    if (!enc.encodeFromDevice((unsigned char*)enc_nv12, (int)enc_pitch, frame_no++, epkts)) {
      return false;
    }
    // NVDEC-кадр больше не нужен: encodeFromDevice уже скопировал его
    // в буфер энкодера (DtoD-копия синхронная). Разблокируем сразу,
    // иначе поверхность декодера висит до конца прогона (locked frames).
    dec.unlock(f.dev_ptr);
    unlock.release();
    if (muxed) {
      std::string err;
      if (!muxer.write(epkts, err)) {
        fprintf(stderr, "[mux] write fail: %s\n", err.c_str());
        return false;
      }
    } else {
      for (auto& e : epkts) {
        if (!WriteAll(fout, e.data.data(), e.data.size())) {
          fprintf(stderr, "[raw] write fail: %s\n", cfg.out_path.c_str());
          return false;
        }
      }
    }
    return true;
  };
  auto guarded = [&](DecodedFrame& f) {
    try {
      return processFrame(f);
    } catch (const std::exception& e) {
      fprintf(stderr, "[fatal-stage] %s\n", e.what());
      return false;
    } catch (...) {
      fprintf(stderr, "[fatal-stage] unknown\n");
      return false;
    }
  };
  bool ok = true;
  bool fatal = false;
  auto t_start = std::chrono::steady_clock::now();

  // Прогресс: строка key=value, её же разбирает GUI. total — оценка:
  // nb_frames, иначе duration*fps; для MKV с аудио-хвостом возможна
  // погрешность в доли процента.
  int64_t progress_total = demux.estimatedFrames();
  if (cfg.max_frames > 0) {
    progress_total = progress_total > 0
        ? std::min<int64_t>(progress_total, cfg.max_frames)
        : (int64_t)cfg.max_frames;
  }
  double last_progress = -1e9;
  auto print_progress = [&](bool force) {
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    if (!force && elapsed - last_progress < 2.0) return;
    last_progress = elapsed;
    const double fps_avg = elapsed > 0 ? (double)frame_no / elapsed : 0.0;
    const double speed = fps > 0 ? fps_avg / fps : 0.0;
    if (progress_total > 0) {
      double pct = 100.0 * (double)frame_no / (double)progress_total;
      if (pct > 100.0) pct = 100.0;
      if (force) pct = 100.0;
      char eta_buf[32];
      if (force) {
        std::snprintf(eta_buf, sizeof(eta_buf), "0.0");
      } else if (fps_avg > 0) {
        const double eta = frame_no < progress_total
            ? (double)(progress_total - frame_no) / fps_avg : 0.0;
        std::snprintf(eta_buf, sizeof(eta_buf), "%.1f", eta);
      } else {
        std::snprintf(eta_buf, sizeof(eta_buf), "?");
      }
      if (cfg.batch_total > 1) {
        std::printf("[progress] frames=%lld total=%lld pct=%.1f fps=%.1f "
                    "speed=%.2fx elapsed=%.1f eta=%s file=%d files=%d\n",
                    (long long)frame_no, (long long)progress_total, pct, fps_avg,
                    speed, elapsed, eta_buf, cfg.batch_index, cfg.batch_total);
      } else {
        std::printf("[progress] frames=%lld total=%lld pct=%.1f fps=%.1f "
                    "speed=%.2fx elapsed=%.1f eta=%s\n",
                    (long long)frame_no, (long long)progress_total, pct, fps_avg,
                    speed, elapsed, eta_buf);
      }
    } else {
      if (cfg.batch_total > 1) {
        std::printf("[progress] frames=%lld total=? pct=? fps=%.1f "
                    "speed=%.2fx elapsed=%.1f eta=? file=%d files=%d\n",
                    (long long)frame_no, fps_avg, speed, elapsed, cfg.batch_index,
                    cfg.batch_total);
      } else {
        std::printf("[progress] frames=%lld total=? pct=? fps=%.1f "
                    "speed=%.2fx elapsed=%.1f eta=?\n",
                    (long long)frame_no, fps_avg, speed, elapsed);
      }
    }
    std::fflush(stdout);
  };
  print_progress(false); // 0% сразу: GUI/лог не висят пустыми
  // При остановке по --frames/ошибке в пачке декодированных кадров часть
  // остаётся locked; NVDEC тогда ругается "locked frames were not unlocked".
  auto unlock_from = [&](size_t from) {
    for (size_t j = from; j < frames.size(); ++j) dec.unlock(frames[j].dev_ptr);
  };
  while (ok && demux.readNext(pkt)) {
    if (!dec.decodePacket(pkt.data.data(), pkt.data.size(), pkt.pts, frames)) { fatal = true; break; }
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (cfg.max_frames > 0 && frame_no >= cfg.max_frames) {
        unlock_from(fi);
        ok = false;
        break;
      }
      if (!guarded(frames[fi])) {
        fatal = true;
        ok = false;
        unlock_from(fi + 1);
        break;
      }
      print_progress(false);
    }
  }
  // Дрейн парсера: запоздалые B-кадры идут тем же путем (иначе warning
  // "locked frames were not unlocked" и потеря хвоста).
  if (ok && !dec.flush(frames)) {
    unlock_from(0); // частично выданные кадры тоже надо вернуть декодеру
    fatal = true;   // хвост не разобран — файл неполный
    ok = false;
  }
  if (ok) {
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (cfg.max_frames > 0 && frame_no >= cfg.max_frames) {
        unlock_from(fi);
        ok = false;
        break;
      }
      if (!guarded(frames[fi])) {
        fatal = true;
        ok = false;
        unlock_from(fi + 1);
        break;
      }
      print_progress(false);
    }
  }
  if (!fatal) print_progress(true);
  if (!enc.flush(epkts)) fatal = true; // хвост потерян — файл не считается успешным
  if (muxed) {
    std::string err;
    if (!muxer.write(epkts, err)) {
      fprintf(stderr, "[mux] flush write fail: %s\n", err.c_str());
      fatal = true;
    }
  } else {
    for (auto& e : epkts) {
      if (!WriteAll(fout, e.data.data(), e.data.size())) {
        fprintf(stderr, "[raw] flush write fail: %s\n", cfg.out_path.c_str());
        fatal = true;
        break;
      }
    }
  }
  // Порядок teardown важен: сначала гасим обертки со своими driver-контекстами
  // и TRT-контекст, и только потом primary CUDA-контекст. Иначе TRT сыпет
  // "invalid device context" при разрушении после cu.shutdown().
  const int dec_errors = dec.decodeErrors();
  enc.shutdown();
  trt.shutdown();
  dec.shutdown();

  if (muxed) {
    std::printf("[mux] remux -> %s\n", cfg.out_path.c_str());
    std::fflush(stdout);
    std::string err;
    if (!muxer.finish(err)) {
      fprintf(stderr, "[mux] finish fail: %s\n", err.c_str());
      fatal = true;
    }
  } else if (fout) {
    if (std::fclose(fout) != 0) {
      fprintf(stderr, "[raw] close fail: %s\n", cfg.out_path.c_str());
      fatal = true;
    }
    fout = nullptr;
  }
  free_dev();
  cu.shutdown();
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  if (dec_errors > 0) {
    fprintf(stderr, "[dec] ошибок декодирования: %d; в видео возможны пропуски кадров\n",
            dec_errors);
  }
  printf("%s, frames=%lld, %.3f s, %.2f fps -> %s\n",
         fatal ? "failed" : "done",
         (long long)frame_no, secs, secs > 0 ? frame_no / secs : 0.0, cfg.out_path.c_str());
  return !fatal;
}
