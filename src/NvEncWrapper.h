#pragma once
// NVENC через SDK NvEncoderCuda. Вход — NV12 или P010 device-ptr (после
// post-kernel), выход — Annex-B/HVCC битстрим.
// Профиль quality повторяет NVEncC: P7 quality, HQ, CQP 18, GOP 250,
// B=5, ref=8 (DPB), lookahead 16, bref-mode middle.
// --split-enc forced_2 делит кадр между двумя NVENC-движками одной сессии.
// DtoD-копия во входной буфер энкодера — ~0.2мс, хоста нет.
#include <cstdint>
#include <string>
#include <vector>

struct EncPacket {
  std::vector<uint8_t> data;
  int64_t pts = 0;
  bool key = false;
};

// Параметры сессии NVENC. Имена повторяют NVEncC, чтобы настройки SVFI
// переносились один-в-один: --preset quality, --bframes 5, --ref 8,
// --lookahead 16, --bref-mode middle, --gop-len 250, --cqp 18.
struct EncoderConfig {
  std::string codec = "h264";   // h264 | hevc | av1
  std::string preset = "p1";    // p1..p7; quality = p7
  std::string tuning = "ull";   // ull | ll | hq (hq = NVEncC quality)
  std::string rc = "cbr";       // cbr | cqp
  int bitrate_kbps = 25000;
  int gop = 120;
  int qp = 18;
  int bframes = 0;
  int refs = 0;                 // 0 = авто; DPB, NVEncC --ref (SVFI: 8)
  int lookahead = 0;            // 0 = выкл; 1..31-bframes
  bool bref_middle = false;     // NV_ENC_BFRAME_REF_MODE_MIDDLE
  int bit_depth = 8;            // 8 = NV12, 10 = P010 / HEVC Main10
  // Split Encoding: разбиение кадра между NVENC-движками одной сессией.
  // auto | auto_forced | forced_2 | forced_3 | disable (H.264 не поддерживает).
  std::string split_enc = "disable";
  int num_encoders = 1;         // 1..2; B-кадры и lookahead требуют 1
};

class NvEncWrapper {
public:
  // num_encoders: 1 или 2. На 5070 Ti (2x Gen9) ставь 2 для 4K60.
  // Режим 2 сессий — chunked GOP: четные GOP на enc0, нечетные на enc1.
  // С B-кадрами/lookahead chunked-режим несовместим (хвост чанка живет
  // в очереди энкодера до следующего кадра) — код принудительно ставит 1.
  bool init(int gpu_id, int w, int h, int fps_num, int fps_den, const EncoderConfig& cfg);
  bool encodeFromDevice(unsigned char* nv12_dev, int pitch, int64_t pts,
                        std::vector<EncPacket>& out);
  bool flush(std::vector<EncPacket>& out);
  int numEncoders() const;
  void shutdown();
  ~NvEncWrapper() { shutdown(); }
private:
  bool drainContiguous(std::vector<EncPacket>& out);
  struct Impl;
  Impl* impl_ = nullptr;
};
