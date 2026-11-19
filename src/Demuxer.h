#pragma once
// Demux только. Никакого декода тут: отдаем сжатые пакеты дальше.
// Под Windows файл открываем с большим буфером, чтобы IO не дергал decode.
#include <cstdint>
#include <string>
#include <vector>

struct Packet {
  std::vector<uint8_t> data;
  int64_t pts = 0;
  bool key = false;
};

class Demuxer {
public:
  bool open(const std::string& path, int* out_w = nullptr, int* out_h = nullptr,
            std::string* out_codec = nullptr, double* out_fps = nullptr,
            int* out_fps_num = nullptr, int* out_fps_den = nullptr);
  bool readNext(Packet& pkt); // false = EOF
  // Начало таймлайна исходника (AV_TIME_BASE, микросекунды): минимум по
  // потокам и PTS первого видеокадра. Нужны муксеру, чтобы аудио/субтитры
  // остались синхронны видео, которое пайплайн начинает с нуля.
  int64_t startTimeUs() const;
  int64_t videoStartTimeUs() const;
  // Оценка общего числа видеокадров: nb_frames, иначе duration*fps.
  // 0 = неизвестно.
  int64_t estimatedFrames() const;
  void close();
  ~Demuxer() { close(); }
private:
  struct Impl;
  Impl* impl_ = nullptr;
};
