#pragma once
// Упаковка в MP4/MKV: видео пишем во временный MPEG-TS (libavformat),
// затем FFmpeg remux-ит TS + исходник в финальный контейнер.
// Так не нужен ручной hvcC/extradata и сохраняются PTS/DTS B-кадров.
#include "NvEncWrapper.h"
#include <string>
#include <vector>

struct MuxConfig {
  std::string out_path;     // .mkv или .mp4
  std::string source_path;  // исходник для аудио/субтитров
  std::string codec;        // hevc | h264 | av1
  int width = 0, height = 0;
  int fps_num = 24000, fps_den = 1001;
  int bit_depth = 8;
  bool copy_audio = true;
  bool copy_subs = true;
  // --frames N: обрезать аудио/субтитры по последнему кадру. Для полного
  // прогона не трогаем хвост исходника (там бывает ещё несколько секунд аудио).
  bool trim_to_video = false;
  // Начало таймлайна исходника (AV_TIME_BASE): база для сдвига аудио/субтитров
  // и PTS первого видеокадра. 0, если исходник начинается с нуля.
  int64_t source_start_us = 0;
  int64_t source_video_start_us = 0;
};

class Muxer {
public:
  bool open(const MuxConfig& cfg, std::string& error);
  bool write(const std::vector<EncPacket>& packets, std::string& error);
  bool finish(std::string& error);
  void abort();
  ~Muxer();
private:
  bool startPending(std::string& error);
  struct Impl;
  Impl* impl_ = nullptr;
};
