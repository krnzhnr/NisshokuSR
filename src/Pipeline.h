#pragma once
// Конфиг и исполнитель одного файла. CLI-слой (Cli.h) заполняет PipelineConfig,
// Pipeline ничего не знает про аргументы командной строки.
#include <string>

struct PipelineConfig {
  std::string in_path, engine_path, out_path;
  std::string onnx_path; // ONNX для сборки engine, если готовый не подошёл
  bool rebuild_engine = false; // игнорировать кэш/готовый engine, собрать из ONNX
  int batch_index = 0; // 1..N для строки прогресса (0 = одиночный файл)
  int batch_total = 0;
  std::string codec = "h264"; // кодек входа для NVDEC (определяет Demuxer)
  std::string out_codec = "h264"; // кодек выхода NVENC: h264/hevc/av1
  int scale = 2;
  int bitrate_kbps = 25000;
  int gop = 120;
  int num_encoders = 2; // 5070 Ti: 2x Gen9 -> ставь 2 для 4K
  bool live_drop = false; // true = дропать опоздавшие, false = блок (файл)
  bool gray = false; // диагностика: вместо compute лить серый кадр
  int max_frames = 0; // диагностика: стоп после N кадров (0 = без лимита)
  std::string tune = "ull"; // hq | ll | ull
  std::string rc = "cbr"; // cbr | cqp
  int qp = 0; // const QP для --rc cqp (напр. 18 как у SVFI)
  // Настройки NVENC как у SVFI/NVEncC; см. EncoderConfig.
  std::string preset = "p1"; // p1..p7, quality = p7
  int bframes = 0;
  int refs = 0; // 0 = авто
  int lookahead = 0; // 0 = выкл
  bool bref_middle = false;
  std::string split_enc = "disable"; // auto | auto_forced | forced_2 | forced_3 | disable
  int bit_depth = 8; // 8 = NV12, 10 = P010 / HEVC Main10
  bool copy_audio = true; // для .mkv/.mp4: копировать аудио из исходника
  bool copy_subs = true;  // для .mkv: копировать субтитры (MP4 их пропускает)
};

class Pipeline {
public:
  bool run(const PipelineConfig& cfg);
};
