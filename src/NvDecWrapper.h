#pragma once
// Тонкая обертка над SDK 13.x NvDecoder (Samples/NvCodec/NvDecoder/NvDecoder.h).
// Декодирует в device-память: 8-bit -> NV12, 10-bit -> P016. Хоста в цикле нет.
// API 13.x: Decode() возвращает число готовых кадров, кадры забираются
// через GetLockedFrame(), возврат — UnlockFrame(&ptr, 1).
#include <cstdint>
#include <string>
#include <vector>

struct DecodedFrame {
  unsigned char* dev_ptr = nullptr; // NV12/P016 device pointer (валиден до unlock)
  int pitch = 0;
  int w = 0, h = 0;
  int bit_depth = 8; // 8 = NV12, 10 = P016
  int64_t pts = 0;
};

class NvDecWrapper {
public:
  bool init(int gpu_id, const std::string& codec /*h264/hevc/av1/vp9*/);
  // Скормить один сжатый пакет. Возвращает кадры через out (0..n из-за B-frame).
  bool decodePacket(const uint8_t* data, size_t size, int64_t pts,
                    std::vector<DecodedFrame>& out);
  bool flush(std::vector<DecodedFrame>& out);
  void unlock(unsigned char* dev_ptr); // вернуть поверхность декодеру
  int decodeErrors() const;            // битые пакеты, пропущенные декодером
  void shutdown();
  ~NvDecWrapper() { shutdown(); }
private:
  struct Impl;
  Impl* impl_ = nullptr;
};
