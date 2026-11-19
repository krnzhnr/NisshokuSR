#pragma once
// TensorRT-апскейлер: готовый engine или сборка из ONNX под текущую GPU/разрешение.
// Кэш движков лежит рядом с ONNX; при несовпадении шейпа/GPU движок пересобирается.
#include <cstddef>
#include <string>

struct TrtUpscaleOptions {
  std::string engine_path; // явный .engine (может быть пустым)
  std::string onnx_path;   // ONNX-модель для сборки (может быть пустой)
  int in_w = 0;
  int in_h = 0;
  int scale = 2;           // ожидаемый множитель; фактический берётся из engine
  bool rebuild = false;    // игнорировать кэш и собрать заново
};

class TrtUpscaler {
public:
  bool init(const TrtUpscaleOptions& opt, std::string& error);
  // dev_in: RGB planar FP16 NCHW (in_w*in_h*3*2 байт), dev_out: RGB planar FP16.
  // stream: comp_stream из CudaContext. Синхронизация — на вызывающем.
  bool inferAsync(void* dev_in, void* dev_out, void* stream);
  int outW() const { return out_w_; }
  int outH() const { return out_h_; }
  size_t inBytes() const;      // данные + slack под OOB-чтение TRT
  size_t outBytes() const;     // данные + slack под OOB-чтение/запись
  size_t inDataBytes() const;  // только RGB planar FP16
  size_t outDataBytes() const; // только RGB planar FP16
  void shutdown();
  ~TrtUpscaler() { shutdown(); }

private:
  bool loadSerialized(const void* data, size_t size, const std::string& source,
                      std::string& error);
  bool loadFile(const std::string& path, std::string& error);

  struct Impl;
  Impl* impl_ = nullptr;
  int in_w_ = 0, in_h_ = 0, out_w_ = 0, out_h_ = 0;
};
