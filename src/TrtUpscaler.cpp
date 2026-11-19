// TensorRT 10.x: готовый engine или сборка из ONNX под текущую GPU и разрешение.
// Кэш: <onnx-stem>_<W>x<H>_<gpu>_fp16_trt<major.minor>.engine рядом с ONNX.
// Сборка требует nvonnxparser_10.dll и nvinfer_builder_resource_10.dll.
#include "TrtUpscaler.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using namespace nvinfer1;
namespace fs = std::filesystem;

namespace {

constexpr size_t kIoSlack = 8 << 20;

// --rebuild-engine в пакете: одна принудительная сборка на ключ
// (ONNX + разрешение + GPU), а не на каждый файл.
std::mutex g_rebuilt_mu;
std::set<std::string> g_rebuilt_keys;

std::string TrtTag() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR);
}

class TrtLogger : public ILogger {
public:
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kERROR) {
      std::lock_guard<std::mutex> lock(mu_);
      last_ = msg ? msg : "";
    }
    if (s <= Severity::kWARNING) {
      std::printf("[TRT] %s\n", msg ? msg : "");
      std::fflush(stdout);
    }
  }
  std::string last() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_;
  }

private:
  mutable std::mutex mu_;
  std::string last_;
};

std::string SanitizeTag(const std::string& s) {
  std::string out;
  bool prev_sep = false;
  for (unsigned char c : s) {
    if (std::isalnum(c)) {
      out.push_back((char)std::tolower(c));
      prev_sep = false;
    } else if (!prev_sep) {
      out.push_back('_');
      prev_sep = true;
    }
  }
  while (!out.empty() && out.back() == '_') out.pop_back();
  return out.empty() ? "gpu" : out;
}

bool GpuInfo(std::string& tag, std::string& error) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    error = "cudaGetDevice failed";
    return false;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    error = "cudaGetDeviceProperties failed";
    return false;
  }
  tag = SanitizeTag(prop.name) + "_cc" + std::to_string(prop.major) +
        std::to_string(prop.minor);
  return true;
}

std::string EngineFileName(const fs::path& onnx, int w, int h,
                           const std::string& gpu_tag) {
  return onnx.stem().u8string() + "_" + std::to_string(w) + "x" + std::to_string(h) +
         "_" + gpu_tag + "_fp16_trt" + TrtTag() + ".engine";
}

fs::path CachePath(const fs::path& onnx, int w, int h, const std::string& gpu_tag) {
  return onnx.parent_path() / fs::u8path(EngineFileName(onnx, w, h, gpu_tag));
}

fs::path FallbackCachePath(const fs::path& onnx, int w, int h,
                           const std::string& gpu_tag) {
  wchar_t local[MAX_PATH];
  const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH);
  fs::path base = (n > 0 && n < MAX_PATH) ? fs::path(local) : fs::temp_directory_path();
  return base / "NisshokuSR" / "engines" / fs::u8path(EngineFileName(onnx, w, h, gpu_tag));
}

// parseFromFile принимает narrow-путь. На Windows это ACP, а не UTF-8,
// поэтому кириллические папки конвертируем через wide с проверкой round-trip.
bool ParserPath(const std::string& utf8, std::string& out, std::string& error) {
  bool ascii = true;
  for (unsigned char c : utf8) {
    if (c >= 0x80) { ascii = false; break; }
  }
  if (ascii) { out = utf8; return true; }

  int wn = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
  if (wn <= 0) { error = "не удалось разобрать UTF-8 путь к ONNX"; return false; }
  std::wstring wide((size_t)wn, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), wn);
  wide.resize((size_t)wn - 1);

  int an = WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (an <= 0) { error = "путь к ONNX не представим в ANSI-кодировке"; return false; }
  std::string acp((size_t)an, '\0');
  WideCharToMultiByte(CP_ACP, 0, wide.c_str(), -1, acp.data(), an, nullptr, nullptr);
  acp.resize((size_t)an - 1);

  int back = MultiByteToWideChar(CP_ACP, 0, acp.c_str(), -1, nullptr, 0);
  std::wstring check((size_t)(back > 0 ? back : 0), L'\0');
  if (back <= 0) { error = "путь к ONNX не представим в ANSI-кодировке"; return false; }
  MultiByteToWideChar(CP_ACP, 0, acp.c_str(), -1, check.data(), back);
  check.resize((size_t)back - 1);
  if (check != wide) {
    error = "путь к ONNX содержит символы вне текущей ANSI-кодировки; "
            "переложите модель в папку с латинским именем";
    return false;
  }
  out = acp;
  return true;
}

std::string CanonicalUtf8(const fs::path& p) {
  std::error_code ec;
  fs::path c = fs::weakly_canonical(p, ec);
  return (ec ? p : c).u8string();
}

long long FileMtime(const fs::path& p) {
  std::error_code ec;
  auto t = fs::last_write_time(p, ec);
  if (ec) return 0;
  return (long long)t.time_since_epoch().count();
}

std::string MetaText(const fs::path& onnx, int w, int h, const std::string& gpu_tag) {
  std::error_code ec;
  const uintmax_t size = fs::file_size(onnx, ec);
  std::string s;
  s += "onnx=" + CanonicalUtf8(onnx) + "\n";
  s += "size=" + std::to_string(ec ? 0 : size) + "\n";
  s += "mtime=" + std::to_string(FileMtime(onnx)) + "\n";
  s += "input=" + std::to_string(w) + "x" + std::to_string(h) + "\n";
  s += "gpu=" + gpu_tag + "\n";
  s += "trt=" + TrtTag() + "\n";
  s += "io=fp16\n";
  return s;
}

// false — метаданные есть, но engine устарел (изменился ONNX/GPU/TRT).
bool MetaMatches(const fs::path& meta, const fs::path& onnx, int w, int h,
                 const std::string& gpu_tag) {
  std::error_code ec;
  if (!fs::is_regular_file(meta, ec)) return true; // метаданных нет — пробуем engine
  std::ifstream f(meta, std::ios::binary);
  if (!f) return false;
  const std::string want = MetaText(onnx, w, h, gpu_tag);
  std::string got((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return got == want;
}

bool SaveEngine(const fs::path& path, const std::vector<char>& blob,
                const std::string& meta, std::string& error) {
  std::error_code ec;
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path(), ec);

  fs::path tmp = path;
  tmp += ".tmp." + std::to_string(GetCurrentProcessId());
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      error = "не удалось открыть на запись " + tmp.u8string();
      return false;
    }
    f.write(blob.data(), (std::streamsize)blob.size());
    f.close();
    if (!f) {
      fs::remove(tmp, ec);
      error = "ошибка записи " + tmp.u8string();
      return false;
    }
  }
  fs::remove(path, ec);
  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(tmp, ec);
    error = "не удалось переименовать " + tmp.u8string() + " -> " + path.u8string();
    return false;
  }
  if (!meta.empty()) {
    std::ofstream m(fs::u8path(path.u8string() + ".meta"),
                    std::ios::binary | std::ios::trunc);
    if (m) m << meta;
  }
  return true;
}

bool ParseOnnx(const std::string& onnx_path, int in_w, int in_h,
               std::vector<char>& blob, std::string& error) {
  TrtLogger logger;
  IBuilder* builder = createInferBuilder(logger);
  if (!builder) { error = "createInferBuilder failed"; return false; }
  auto cleanup = [&](INetworkDefinition* net, IBuilderConfig* cfg, nvonnxparser::IParser* parser) {
    if (parser) delete parser;
    if (cfg) delete cfg;
    if (net) delete net;
    if (builder) delete builder;
  };

  INetworkDefinition* network = builder->createNetworkV2(0U);
  if (!network) {
    cleanup(nullptr, nullptr, nullptr);
    error = "createNetworkV2 failed";
    return false;
  }
  nvonnxparser::IParser* parser =
      nvonnxparser::createParser(*network, logger);
  if (!parser) {
    cleanup(network, nullptr, nullptr);
    error = "createParser failed (nvonnxparser_10.dll рядом с exe?)";
    return false;
  }

  std::string parser_path;
  if (!ParserPath(onnx_path, parser_path, error)) {
    cleanup(network, nullptr, parser);
    return false;
  }

  const bool parsed = parser->parseFromFile(parser_path.c_str(),
                                            (int)ILogger::Severity::kWARNING);
  if (!parsed) {
    error = "ONNX не разобран: " + onnx_path;
    const int n = parser->getNbErrors();
    for (int i = 0; i < n && i < 4; ++i) {
      const auto* e = parser->getError(i);
      if (!e) continue;
      error += "\n  ";
      if (e->file() && *e->file()) {
        error += std::string(e->file()) + ":" + std::to_string(e->line()) + ": ";
      }
      error += e->desc() ? e->desc() : "unknown error";
    }
    if (n == 0) {
      const std::string last = logger.last();
      if (!last.empty()) error += "\n  " + last;
    }
    cleanup(network, nullptr, parser);
    return false;
  }

  if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
    error = "ожидается ровно 1 вход и 1 выход ONNX, получено " +
            std::to_string(network->getNbInputs()) + "/" +
            std::to_string(network->getNbOutputs());
    cleanup(network, nullptr, parser);
    return false;
  }

  ITensor* input = network->getInput(0);
  ITensor* output = network->getOutput(0);
  const Dims in_dims = input->getDimensions();
  const Dims out_dims = output->getDimensions();
  if (out_dims.nbDims != 4) {
    error = "выход ONNX должен быть NCHW (4 измерения)";
    cleanup(network, nullptr, parser);
    return false;
  }
  if (out_dims.d[1] > 0 && out_dims.d[1] != 3) {
    error = "ожидается 3 канала RGB на выходе ONNX";
    cleanup(network, nullptr, parser);
    return false;
  }
  if (in_dims.nbDims != 4) {
    error = "вход ONNX должен быть NCHW (4 измерения)";
    cleanup(network, nullptr, parser);
    return false;
  }
  if (in_dims.d[0] > 0 && in_dims.d[0] != 1) {
    error = "батч входа ONNX должен быть 1 или динамическим";
    cleanup(network, nullptr, parser);
    return false;
  }
  if (in_dims.d[1] > 0 && in_dims.d[1] != 3) {
    error = "ожидается 3 канала RGB на входе ONNX";
    cleanup(network, nullptr, parser);
    return false;
  }

  bool dynamic = false;
  for (int i = 0; i < in_dims.nbDims; ++i) {
    if (in_dims.d[i] < 0) dynamic = true;
  }
  if (!dynamic) {
    if (in_dims.d[2] != in_h || in_dims.d[3] != in_w) {
      error = "ONNX имеет статический вход " + std::to_string(in_dims.d[3]) + "x" +
              std::to_string(in_dims.d[2]) + ", а видео " + std::to_string(in_w) + "x" +
              std::to_string(in_h) +
              ". Экспортируйте модель с --h/--w под это разрешение "
              "или используйте динамический вход";
      cleanup(network, nullptr, parser);
      return false;
    }
  }

  // Пайплайн всегда работает с FP16 IO. Для FP32-модели принудительно
  // переводим вход/выход на half: builder вставляет reformat/cast.
  if (input->getType() != DataType::kHALF) input->setType(DataType::kHALF);
  if (output->getType() != DataType::kHALF) output->setType(DataType::kHALF);

  IBuilderConfig* config = builder->createBuilderConfig();
  if (!config) {
    cleanup(network, nullptr, parser);
    error = "createBuilderConfig failed";
    return false;
  }
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, (size_t)4 << 30);
  config->setBuilderOptimizationLevel(5);
  config->setFlag(BuilderFlag::kFP16);

  if (dynamic) {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (!profile) {
      cleanup(network, config, parser);
      error = "createOptimizationProfile failed";
      return false;
    }
    const Dims concrete{4, {1, 3, (int64_t)in_h, (int64_t)in_w}};
    const char* name = input->getName();
    profile->setDimensions(name, OptProfileSelector::kMIN, concrete);
    profile->setDimensions(name, OptProfileSelector::kOPT, concrete);
    profile->setDimensions(name, OptProfileSelector::kMAX, concrete);
    if (config->addOptimizationProfile(profile) < 0) {
      cleanup(network, config, parser);
      error = "addOptimizationProfile failed";
      return false;
    }
  }

  std::printf("[engine] сборка TensorRT %s: %s, вход %dx%d...\n",
              TrtTag().c_str(), onnx_path.c_str(), in_w, in_h);
  std::fflush(stdout);

  IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
  if (!mem) {
    error = "buildSerializedNetwork failed";
    const std::string last = logger.last();
    if (!last.empty()) error += ": " + last;
    cleanup(network, config, parser);
    return false;
  }
  blob.assign((const char*)mem->data(), (const char*)mem->data() + mem->size());
  delete mem;
  cleanup(network, config, parser);
  return true;
}

} // namespace

struct TrtUpscaler::Impl {
  std::unique_ptr<TrtLogger> logger;
  IRuntime* runtime = nullptr;
  ICudaEngine* engine = nullptr;
  IExecutionContext* ctx = nullptr;
  std::string in_name, out_name;
  void* bound_in = nullptr;
  void* bound_out = nullptr;
};

bool TrtUpscaler::loadFile(const std::string& path, std::string& error) {
  std::ifstream f(fs::u8path(path), std::ios::binary);
  if (!f) {
    error = "не удалось открыть " + path;
    return false;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
  if (blob.empty()) {
    error = "пустой engine: " + path;
    return false;
  }
  return loadSerialized(blob.data(), blob.size(), path, error);
}

bool TrtUpscaler::loadSerialized(const void* data, size_t size,
                                 const std::string& source, std::string& error) {
  shutdown();
  out_w_ = out_h_ = 0;
  impl_ = new Impl();
  impl_->logger = std::make_unique<TrtLogger>();
  impl_->runtime = createInferRuntime(*impl_->logger);
  if (!impl_->runtime) {
    error = "createInferRuntime failed";
    shutdown();
    return false;
  }
  impl_->engine = impl_->runtime->deserializeCudaEngine(data, size);
  if (!impl_->engine) {
    error = "engine не загрузился (" + source + "): " + impl_->logger->last();
    shutdown();
    return false;
  }

  int n_in = 0, n_out = 0;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char* nm = impl_->engine->getIOTensorName(i);
    const TensorIOMode mode = impl_->engine->getTensorIOMode(nm);
    if (mode == TensorIOMode::kINPUT) {
      ++n_in;
      impl_->in_name = nm;
    } else if (mode == TensorIOMode::kOUTPUT) {
      ++n_out;
      impl_->out_name = nm;
    }
  }
  if (n_in != 1 || n_out != 1) {
    error = "в engine ожидается 1 вход и 1 выход, найдено " + std::to_string(n_in) +
            "/" + std::to_string(n_out);
    shutdown();
    return false;
  }

  const DataType in_type = impl_->engine->getTensorDataType(impl_->in_name.c_str());
  const DataType out_type = impl_->engine->getTensorDataType(impl_->out_name.c_str());
  if (in_type != DataType::kHALF || out_type != DataType::kHALF) {
    error = "engine имеет не-FP16 вход/выход (" + std::to_string((int)in_type) + "/" +
            std::to_string((int)out_type) +
            "); пересоберите модель с FP16 IO (export_onnx.py --dtype fp16)";
    shutdown();
    return false;
  }

  const Dims dims = impl_->engine->getTensorShape(impl_->in_name.c_str());
  if (dims.nbDims != 4) {
    error = "вход engine не NCHW";
    shutdown();
    return false;
  }
  if (dims.d[0] > 0 && dims.d[0] != 1) {
    error = "батч engine должен быть 1";
    shutdown();
    return false;
  }
  if (dims.d[1] > 0 && dims.d[1] != 3) {
    error = "engine ожидает не 3 канала";
    shutdown();
    return false;
  }
  bool dynamic = false;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) dynamic = true;
  }
  if (!dynamic) {
    if (dims.d[2] != in_h_ || dims.d[3] != in_w_) {
      error = "engine собран под " + std::to_string(dims.d[3]) + "x" +
              std::to_string(dims.d[2]) + ", а нужно " + std::to_string(in_w_) + "x" +
              std::to_string(in_h_);
      shutdown();
      return false;
    }
  }

  impl_->ctx = impl_->engine->createExecutionContext();
  if (!impl_->ctx) {
    error = "createExecutionContext failed: " + impl_->logger->last();
    shutdown();
    return false;
  }
  if (dynamic) {
    const Dims concrete{4, {1, 3, (int64_t)in_h_, (int64_t)in_w_}};
    if (!impl_->ctx->setInputShape(impl_->in_name.c_str(), concrete)) {
      error = "setInputShape(" + std::to_string(in_w_) + "x" + std::to_string(in_h_) +
              ") failed: разрешение вне профиля engine";
      shutdown();
      return false;
    }
  }

  const Dims out = impl_->ctx->getTensorShape(impl_->out_name.c_str());
  if (out.nbDims != 4 || out.d[0] != 1 || out.d[1] != 3 || out.d[2] <= 0 || out.d[3] <= 0) {
    error = "не удалось определить выход engine (нужен статический NCHW 1x3xHxW)";
    shutdown();
    return false;
  }
  out_h_ = (int)out.d[2];
  out_w_ = (int)out.d[3];
  return true;
}

bool TrtUpscaler::init(const TrtUpscaleOptions& opt, std::string& error) {
  shutdown();
  error.clear();
  in_w_ = opt.in_w;
  in_h_ = opt.in_h;
  out_w_ = out_h_ = 0;

  if (in_w_ <= 0 || in_h_ <= 0) {
    error = "некорректное разрешение входа";
    return false;
  }
  if (opt.engine_path.empty() && opt.onnx_path.empty()) {
    error = "не задан ни TensorRT engine (--engine), ни ONNX-модель (--onnx)";
    return false;
  }
  if (opt.rebuild && opt.onnx_path.empty()) {
    error = "--rebuild-engine требует --onnx";
    return false;
  }

  std::string gpu_tag;
  if (!GpuInfo(gpu_tag, error)) return false;

  auto loaded = [&](const fs::path& p, const char* what) -> bool {
    std::string err;
    if (!fs::is_regular_file(p)) return false;
    if (!loadFile(p.u8string(), err)) {
      std::fprintf(stderr, "[engine] %s не подошёл: %s\n", what, err.c_str());
      return false;
    }
    std::printf("[engine] %s: %s\n", what, p.u8string().c_str());
    std::fflush(stdout);
    return true;
  };

  if (!opt.rebuild && !opt.engine_path.empty()) {
    const fs::path p = fs::u8path(opt.engine_path);
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) {
      if (opt.onnx_path.empty()) {
        error = "engine не найден: " + opt.engine_path;
        return false;
      }
      std::fprintf(stderr, "[engine] engine не найден (%s), пересоберу из ONNX\n",
                   opt.engine_path.c_str());
    } else {
      if (loaded(p, "готовый engine")) return true;
      if (opt.onnx_path.empty()) {
        error = "engine не подходит, а ONNX для пересборки не указан: " + opt.engine_path;
        return false;
      }
      std::fprintf(stderr, "[engine] пересоберу из ONNX: %s\n", opt.onnx_path.c_str());
    }
  }

  if (opt.onnx_path.empty()) {
    error = "не удалось загрузить engine и не задан --onnx";
    return false;
  }
  const fs::path onnx = fs::u8path(opt.onnx_path);
  std::error_code ec;
  if (!fs::is_regular_file(onnx, ec)) {
    error = "ONNX не найден: " + opt.onnx_path;
    return false;
  }

  const fs::path cache = CachePath(onnx, in_w_, in_h_, gpu_tag);
  const fs::path fallback = FallbackCachePath(onnx, in_w_, in_h_, gpu_tag);
  const std::string rebuild_key = CanonicalUtf8(onnx) + "|" +
      std::to_string(in_w_) + "x" + std::to_string(in_h_) + "|" + gpu_tag;
  bool force_rebuild = opt.rebuild;
  if (force_rebuild) {
    std::lock_guard<std::mutex> lock(g_rebuilt_mu);
    if (g_rebuilt_keys.count(rebuild_key)) force_rebuild = false;
  }

  if (!force_rebuild) {
    const fs::path sibling =
        onnx.parent_path() / fs::u8path(onnx.stem().u8string() + ".engine");
    if (loaded(sibling, "готовый engine рядом с ONNX")) return true;

    for (const fs::path& candidate : {cache, fallback}) {
      if (!fs::is_regular_file(candidate, ec)) continue;
      if (!MetaMatches(fs::u8path(candidate.u8string() + ".meta"), onnx, in_w_, in_h_,
                       gpu_tag)) {
        std::printf("[engine] кэш устарел (изменился ONNX/GPU/TRT): %s\n",
                    candidate.u8string().c_str());
        continue;
      }
      if (loaded(candidate, "кэш engine")) return true;
    }
  }

  std::vector<char> blob;
  const auto t0 = std::chrono::steady_clock::now();
  if (!ParseOnnx(opt.onnx_path, in_w_, in_h_, blob, error)) return false;
  const double build_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  const std::string meta = MetaText(onnx, in_w_, in_h_, gpu_tag);
  std::string save_error;
  fs::path saved;
  if (SaveEngine(cache, blob, meta, save_error)) {
    saved = cache;
  } else {
    std::fprintf(stderr, "[engine] не удалось сохранить кэш рядом с ONNX: %s\n",
                 save_error.c_str());
    if (SaveEngine(fallback, blob, meta, save_error)) {
      saved = fallback;
    } else {
      std::fprintf(stderr, "[engine] кэш не сохранён: %s\n", save_error.c_str());
    }
  }

  if (!loadSerialized(blob.data(), blob.size(), "новая сборка", error)) return false;
  if (opt.rebuild) {
    std::lock_guard<std::mutex> lock(g_rebuilt_mu);
    g_rebuilt_keys.insert(rebuild_key);
  }
  if (!saved.empty()) {
    std::printf("[engine] собран за %.1f s -> %s\n", build_s, saved.u8string().c_str());
  } else {
    std::printf("[engine] собран за %.1f s (в памяти, без кэша)\n", build_s);
  }
  std::fflush(stdout);
  return true;
}

bool TrtUpscaler::inferAsync(void* dev_in, void* dev_out, void* stream) {
  if (!impl_ || !impl_->ctx) return false;
  if (impl_->bound_in != dev_in || impl_->bound_out != dev_out) {
    impl_->ctx->setTensorAddress(impl_->in_name.c_str(), dev_in);
    impl_->ctx->setTensorAddress(impl_->out_name.c_str(), dev_out);
    impl_->bound_in = dev_in;
    impl_->bound_out = dev_out;
  }
  return impl_->ctx->enqueueV3((cudaStream_t)stream);
}

size_t TrtUpscaler::inDataBytes() const {
  return (size_t)in_w_ * in_h_ * 3 * sizeof(__half);
}
size_t TrtUpscaler::outDataBytes() const {
  return (size_t)out_w_ * out_h_ * 3 * sizeof(__half);
}
size_t TrtUpscaler::inBytes() const {
  return inDataBytes() + kIoSlack;
}
size_t TrtUpscaler::outBytes() const {
  return outDataBytes() + kIoSlack;
}

void TrtUpscaler::shutdown() {
  if (impl_) {
    delete impl_->ctx;
    delete impl_->engine;
    delete impl_->runtime;
    delete impl_;
    impl_ = nullptr;
  }
  out_w_ = out_h_ = 0;
}
