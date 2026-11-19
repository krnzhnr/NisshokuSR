#include "Cli.h"
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <system_error>

#ifndef NISSHOKU_VERSION
#define NISSHOKU_VERSION "0.3.0"
#endif

namespace {

std::string ToLower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

bool IsOneOf(const std::string& v, std::initializer_list<const char*> list) {
  for (const char* x : list) if (v == x) return true;
  return false;
}

bool IsMuxedOut(const std::string& path) {
  const std::string l = ToLower(path);
  const size_t n = l.size();
  return (n >= 4 && l.compare(n - 4, 4, ".mkv") == 0) ||
         (n >= 4 && l.compare(n - 4, 4, ".mp4") == 0);
}

bool HasKnownExt(const std::string& path) {
  const std::string l = ToLower(path);
  for (const char* e : {".mkv", ".mp4", ".h264", ".h265", ".hevc", ".av1"}) {
    const size_t n = std::strlen(e);
    if (l.size() > n && l.compare(l.size() - n, n, e) == 0) return true;
  }
  return false;
}

bool LooksLikeDir(const std::string& path) {
  if (path.empty()) return false;
  const char last = path.back();
  if (last == '/' || last == '\\') return true;
  std::error_code ec;
  return std::filesystem::is_directory(std::filesystem::u8path(path), ec);
}

bool NormalizeOutExt(const std::string& value, const std::string& codec,
                     std::string& out, std::string& error) {
  std::string v = ToLower(value);
  if (!v.empty() && v[0] == '.') v.erase(0, 1);
  if (v == "raw") {
    if (codec == "h264") v = "h264";
    else if (codec == "av1") v = "av1";
    else v = "h265";
  }
  if (v == "hevc") v = "h265";
  if (!IsOneOf(v, {"mp4", "mkv", "h264", "h265", "av1"})) {
    error = "--out-ext: ожидается mp4|mkv|h264|h265|av1|raw";
    return false;
  }
  if ((v == "mp4" || v == "mkv") && codec == "av1") {
    error = "AV1 в MP4/MKV не поддерживается: используйте --out-ext av1 (raw)";
    return false;
  }
  if (v == "h265" && codec != "hevc" && codec != "h265") {
    error = "--out-ext h265 требует --codec-out hevc";
    return false;
  }
  if (v == "h264" && codec != "h264") {
    error = "--out-ext h264 требует --codec-out h264";
    return false;
  }
  if (v == "av1" && codec != "av1") {
    error = "--out-ext av1 требует --codec-out av1";
    return false;
  }
  out = v;
  return true;
}

// Проверка -o <файл>: расширение должно соответствовать кодеку.
// Для .mkv/.mp4 проверяется только запрет AV1; для raw — точное совпадение.
bool ValidateOutPathCodec(const std::string& path, const std::string& codec,
                          std::string& error) {
  const std::string l = ToLower(path);
  auto ends = [&](const char* suffix) {
    const size_t n = std::strlen(suffix);
    return l.size() >= n && l.compare(l.size() - n, n, suffix) == 0;
  };
  if (ends(".mkv") || ends(".mp4")) {
    if (codec == "av1") {
      error = "AV1 в MKV/MP4 пока не поддерживается: используйте raw .av1 или hevc/h264";
      return false;
    }
    return true;
  }
  if (ends(".h265") || ends(".hevc")) {
    if (codec != "hevc" && codec != "h265") {
      error = "выход .h265/.hevc требует --codec-out hevc";
      return false;
    }
    return true;
  }
  if (ends(".h264")) {
    if (codec != "h264") {
      error = "выход .h264 требует --codec-out h264";
      return false;
    }
    return true;
  }
  if (ends(".av1")) {
    if (codec != "av1") {
      error = "выход .av1 требует --codec-out av1";
      return false;
    }
    return true;
  }
  return true; // .raw или файл без расширения — сырой поток, кодек задаёт --codec-out
}

// 0 = значение получено, 1 = опция была, но значения нет, -1 = не наша опция.
int TryValue(const std::vector<std::string>& args, size_t& i, const char* name,
             std::string& out, std::string& error) {
  const std::string n = name;
  const std::string& a = args[i];
  if (a == n) {
    if (i + 1 >= args.size()) {
      error = "опция " + n + " требует значение";
      return 1;
    }
    out = args[++i];
    return 0;
  }
  if (a.size() > n.size() && a.compare(0, n.size(), n) == 0 && a[n.size()] == '=') {
    out = a.substr(n.size() + 1);
    return 0;
  }
  return -1;
}

bool ParseInt(const std::string& s, int lo, int hi, int& out, std::string& error,
              const char* name) {
  if (s.empty()) {
    error = std::string("пустое значение для ") + name;
    return false;
  }
  errno = 0;
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (errno != 0 || end == s.c_str() || *end != '\0' || v < lo || v > hi) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: ожидается число %d..%d, получено '%s'",
                  name, lo, hi, s.c_str());
    error = buf;
    return false;
  }
  out = (int)v;
  return true;
}

bool ApplyProfile(const std::string& name, PipelineConfig& cfg, std::string& error) {
  if (name == "fast") {
    // Быстрый файловый профиль: лёгкий NVENC, без B/lookahead, 2 сессии.
    cfg.out_codec = "hevc";
    cfg.preset = "p1";
    cfg.tune = "hq";
    cfg.rc = "cqp";
    cfg.qp = 18;
    cfg.gop = 250;
    cfg.bframes = 0;
    cfg.refs = 0;
    cfg.lookahead = 0;
    cfg.bref_middle = false;
    cfg.bit_depth = 8;
    cfg.split_enc = "disable";
    cfg.num_encoders = 2;
    return true;
  }
  if (name == "balanced") {
    // P7 без B-кадров: 10-бит, chunked dual.
    cfg.out_codec = "hevc";
    cfg.preset = "p7";
    cfg.tune = "hq";
    cfg.rc = "cqp";
    cfg.qp = 18;
    cfg.gop = 250;
    cfg.bframes = 0;
    cfg.refs = 0;
    cfg.lookahead = 0;
    cfg.bref_middle = false;
    cfg.bit_depth = 10;
    cfg.split_enc = "disable";
    cfg.num_encoders = 2;
    return true;
  }
  if (name == "quality" || name == "quality-split") {
    cfg.out_codec = "hevc";
    cfg.preset = "p7";
    cfg.tune = "hq";
    cfg.rc = "cqp";
    cfg.qp = 18;
    cfg.gop = 250;
    cfg.bframes = 5;
    cfg.refs = 8;
    cfg.lookahead = 16;
    cfg.bref_middle = true;
    cfg.bit_depth = 10;
    cfg.split_enc = (name == "quality-split") ? "forced_2" : "disable";
    cfg.num_encoders = 1; // B-кадры/lookahead несовместимы с chunked dual
    return true;
  }
  error = "неизвестный профиль '" + name + "' (fast|balanced|quality|quality-split)";
  return false;
}

enum class Opt { No, Yes, Err };

bool ParseOption(const std::vector<std::string>& args, size_t& i, PipelineConfig& cfg,
                 bool& dry_run, std::vector<std::string>& inputs,
                 std::string& out_dir, std::string& out_ext, bool& out_ext_set,
                 std::string& error) {
  auto take_str = [&](const char* name, std::string& dst) -> Opt {
    std::string v;
    int r = TryValue(args, i, name, v, error);
    if (r == 0) { dst = v; return Opt::Yes; }
    if (r == 1) return Opt::Err;
    return Opt::No;
  };
  auto take_int = [&](const char* name, int& dst, int lo, int hi) -> Opt {
    std::string v;
    int r = TryValue(args, i, name, v, error);
    if (r == 1) return Opt::Err;
    if (r != 0) return Opt::No;
    if (!ParseInt(v, lo, hi, dst, error, name)) return Opt::Err;
    return Opt::Yes;
  };
  auto done = [](Opt o) { return o == Opt::Yes; };
  auto failed = [](Opt o) { return o == Opt::Err; };

  const std::string& a = args[i];
  if (a == "--dry-run") { dry_run = true; return true; }
  if (a == "--live") { cfg.live_drop = true; return true; }
  if (a == "--gray") { cfg.gray = true; return true; }
  if (a == "--no-audio") { cfg.copy_audio = false; return true; }
  if (a == "--no-subs") { cfg.copy_subs = false; return true; }
  if (a == "--10bit") { cfg.bit_depth = 10; return true; }
  if (a == "--rebuild-engine") { cfg.rebuild_engine = true; return true; }

  {
    std::string v;
    Opt o = take_str("--in", v);
    if (o == Opt::No) o = take_str("-i", v);
    if (done(o)) { inputs.push_back(v); return true; }
    if (failed(o)) return false;
  }

  {
    std::string v;
    Opt o = take_str("--onnx", v);
    if (done(o)) { cfg.onnx_path = v; return true; }
    if (failed(o)) return false;
  }

  {
    std::string v;
    Opt o = take_str("--out-dir", v);
    if (done(o)) { out_dir = v; return true; }
    if (failed(o)) return false;
  }

  {
    std::string v;
    Opt o = take_str("--out-ext", v);
    if (done(o)) { out_ext = v; out_ext_set = true; return true; }
    if (failed(o)) return false;
  }
  {
    std::string v;
    Opt o = take_str("--engine", v);
    if (o == Opt::No) o = take_str("-e", v);
    if (done(o)) { cfg.engine_path = v; return true; }
    if (failed(o)) return false;
  }

  Opt o = take_str("--out", cfg.out_path);
  if (o == Opt::No) o = take_str("-o", cfg.out_path);
  if (done(o)) return true;
  if (failed(o)) return false;

  {
    std::string v;
    o = take_str("--codec-out", v);
    if (done(o)) {
      if (!IsOneOf(v, {"h264", "hevc", "h265", "av1"})) {
        error = "--codec-out: ожидается h264|hevc|h265|av1";
        return false;
      }
      cfg.out_codec = v;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--tune", v);
    if (done(o)) {
      if (!IsOneOf(v, {"ull", "ll", "hq", "quality"})) {
        error = "--tune: ожидается ull|ll|hq";
        return false;
      }
      cfg.tune = v;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--rc", v);
    if (done(o)) {
      if (!IsOneOf(v, {"cbr", "cqp", "constqp"})) {
        error = "--rc: ожидается cbr|cqp";
        return false;
      }
      cfg.rc = v;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--preset", v);
    if (done(o)) {
      if (!IsOneOf(v, {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "quality", "performance"})) {
        error = "--preset: ожидается p1..p7|quality|performance";
        return false;
      }
      cfg.preset = v;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--split-enc", v);
    if (done(o)) {
      if (!IsOneOf(v, {"auto", "auto_forced", "forced_2", "forced_3", "disable"})) {
        error = "--split-enc: ожидается auto|auto_forced|forced_2|forced_3|disable";
        return false;
      }
      cfg.split_enc = v;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    int depth = cfg.bit_depth;
    o = take_int("--bit-depth", depth, 8, 10);
    if (done(o)) {
      if (depth != 8 && depth != 10) {
        error = "--bit-depth: ожидается 8 или 10";
        return false;
      }
      cfg.bit_depth = depth;
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--bref-mode", v);
    if (done(o)) {
      const std::string l = ToLower(v);
      if (l == "middle") cfg.bref_middle = true;
      else if (l == "disabled" || l == "auto") cfg.bref_middle = false;
      else {
        error = "--bref-mode: ожидается middle|disabled|auto";
        return false;
      }
      return true;
    }
    if (failed(o)) return false;
  }

  {
    std::string v;
    o = take_str("--audio", v);
    if (done(o)) {
      if (v == "copy") cfg.copy_audio = true;
      else if (v == "none") cfg.copy_audio = false;
      else { error = "--audio: ожидается copy|none"; return false; }
      return true;
    }
    if (failed(o)) return false;
  }
  {
    std::string v;
    o = take_str("--subs", v);
    if (done(o)) {
      if (v == "copy") cfg.copy_subs = true;
      else if (v == "none") cfg.copy_subs = false;
      else { error = "--subs: ожидается copy|none"; return false; }
      return true;
    }
    if (failed(o)) return false;
  }

  o = take_int("--scale", cfg.scale, 1, 8); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--bitrate", cfg.bitrate_kbps, 1, 1000000); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--gop", cfg.gop, 1, 10000); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--encoders", cfg.num_encoders, 1, 2); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--qp", cfg.qp, 0, 51); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--bframes", cfg.bframes, 0, 5); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--ref", cfg.refs, 0, 16); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--lookahead", cfg.lookahead, 0, 31); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--frames", cfg.max_frames, 0, 1000000000); if (done(o)) return true; if (failed(o)) return false;
  o = take_int("--max-frames", cfg.max_frames, 0, 1000000000); if (done(o)) return true; if (failed(o)) return false;

  // Позиционные аргументы — входные файлы/папки (удобно для drag&drop и массивов).
  if (!a.empty() && a[0] != '-') {
    inputs.push_back(a);
    return true;
  }

  error = a.rfind("-", 0) == 0 ? "неизвестная опция: " + a
                               : "неожиданный аргумент: " + a;
  return false;
}

} // namespace

CliAction ParseCli(const std::vector<std::string>& args, CliOptions& opt, std::string& error) {
  opt = CliOptions{};
  error.clear();
  if (args.size() <= 1) return CliAction::Help;

  // Справка/версия/список профилей важнее остальных аргументов.
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--help" || a == "-h" || a == "-?" || a == "/?") return CliAction::Help;
    if (a == "--version" || a == "-V") return CliAction::Version;
    if (a == "--list-profiles") return CliAction::ListProfiles;
  }

  // Профиль выбираем до разбора остального: явные флаги всегда перекрывают его,
  // независимо от порядка в командной строке.
  std::string profile;
  for (size_t i = 1; i < args.size(); ++i) {
    std::string v;
    int r = TryValue(args, i, "--profile", v, error);
    if (r == 1) return CliAction::Error;
    if (r == 0) {
      if (v.empty()) { error = "пустое имя профиля"; return CliAction::Error; }
      profile = ToLower(v);
      continue;
    }
  }

  PipelineConfig cfg;
  if (!profile.empty()) {
    if (!ApplyProfile(profile, cfg, error)) return CliAction::Error;
    opt.profile = profile;
  }
  opt.cfg = cfg;

  bool out_ext_set = false;
  for (size_t i = 1; i < args.size(); ++i) {
    std::string v;
    int r = TryValue(args, i, "--profile", v, error);
    if (r == 1) return CliAction::Error;
    if (r == 0) continue;
    if (!ParseOption(args, i, opt.cfg, opt.dry_run, opt.inputs, opt.out_dir,
                     opt.out_ext, out_ext_set, error)) {
      return CliAction::Error;
    }
  }

  bool any_input_dir = false;
  for (const std::string& in : opt.inputs) {
    if (LooksLikeDir(in)) { any_input_dir = true; break; }
  }

  if (!opt.cfg.out_path.empty()) {
    if (!opt.out_dir.empty()) {
      error = "нельзя одновременно -o и --out-dir";
      return CliAction::Error;
    }
    if (LooksLikeDir(opt.cfg.out_path)) {
      opt.out_dir = opt.cfg.out_path;
      opt.cfg.out_path.clear();
    } else if (opt.inputs.size() > 1 || any_input_dir) {
      if (HasKnownExt(opt.cfg.out_path)) {
        error = "при нескольких входах или папке на входе -o должен быть папкой";
        return CliAction::Error;
      }
      opt.out_dir = opt.cfg.out_path;
      opt.cfg.out_path.clear();
    } else if (out_ext_set) {
      error = "--out-ext используется только для пакетного выхода; "
              "с -o <файл> формат задаётся расширением";
      return CliAction::Error;
    }
  }

  if (!opt.cfg.out_path.empty() &&
      !ValidateOutPathCodec(opt.cfg.out_path, opt.cfg.out_codec, error)) {
    return CliAction::Error;
  }

  if (opt.inputs.empty() && !opt.dry_run) {
    error = "не указан --in";
    return CliAction::Error;
  }
  if (!opt.dry_run && opt.cfg.engine_path.empty() && opt.cfg.onnx_path.empty()) {
    error = "не указан --engine или --onnx";
    return CliAction::Error;
  }

  if (out_ext_set) {
    if (!NormalizeOutExt(opt.out_ext, opt.cfg.out_codec, opt.out_ext, error)) {
      return CliAction::Error;
    }
  } else {
    opt.out_ext = "mp4";
    if (opt.cfg.out_codec == "av1" && opt.cfg.out_path.empty()) {
      error = "AV1 в MP4 не поддерживается: укажите --out-ext av1";
      return CliAction::Error;
    }
  }

  if (opt.cfg.rebuild_engine && opt.cfg.onnx_path.empty()) {
    error = "--rebuild-engine требует --onnx";
    return CliAction::Error;
  }

  const bool raw_output =
      (!opt.cfg.out_path.empty() && !IsMuxedOut(opt.cfg.out_path)) ||
      (opt.cfg.out_path.empty() && opt.out_ext != "mp4" && opt.out_ext != "mkv");
  if (raw_output && opt.cfg.bframes > 0) {
    error = "raw-выход не поддерживается при B-кадрах: используйте .mkv/.mp4 "
            "или профиль/флаги с --bframes 0";
    return CliAction::Error;
  }

  return CliAction::Run;
}

void PrintVersion() {
  std::printf("nisshoku %s\n", NISSHOKU_VERSION);
}

void PrintProfiles() {
  std::printf(
      "Профили:\n"
      "  fast           P1/HQ, CQP18, B=0, dual NVENC, 8-bit, HEVC\n"
      "  balanced       P7/HQ, CQP18, B=0, dual NVENC, 10-bit, HEVC\n"
      "  quality        P7/HQ, CQP18, B=5, ref=8, LA=16, Main10, 1 сессия\n"
      "  quality-split  quality + split encoding forced_2 (оба NVENC на один поток)\n"
      "\n"
      "Явные флаги перекрывают профиль: --profile quality --bframes 3 --qp 20\n");
}

void PrintHelp() {
  std::printf(
      "nisshoku — NVDEC -> TensorRT -> NVENC, всё на GPU (Windows/NVIDIA)\n"
      "\n"
      "Использование:\n"
      "  nisshoku -i <вход> -e <engine> -o <выход> [опции]\n"
      "  nisshoku -i <вход> --onnx <model.onnx> -o <выход>\n"
      "  nisshoku --profile quality-split -i \"D:\\серии\" -o \"D:\\out\" --onnx model.onnx\n"
      "  nisshoku --profile quality-split a.mkv b.mkv c.mp4 --onnx model.onnx\n"
      "  nisshoku --dry-run --profile quality-split\n"
      "  nisshoku --list-profiles\n"
      "\n"
      "Вход/выход:\n"
      "  -i, --in <path>            файл или папка; можно повторять\n"
      "  <path>...                  позиционные входы (файлы/папки)\n"
      "  -e, --engine <path>        готовый TensorRT engine (необязателен при --onnx)\n"
      "  --onnx <path>              ONNX: готовый engine подхватится, иначе соберётся\n"
      "  --rebuild-engine           игнорировать кэш/engine и собрать заново из ONNX\n"
      "  -o, --out <path>           файл (один вход) или папка (папка/несколько входов)\n"
      "  --out-dir <dir>            папка выхода (по умолчанию <папка входа>\\upscaled)\n"
      "  --out-ext <ext>            mp4|mkv|h264|h265|av1|raw для пакетных имён (mp4)\n"
      "  --codec-out <codec>        h264|hevc|h265|av1 (AV1 — только raw; по умолчанию h264)\n"
      "  --audio <copy|none>        аудио из исходника (по умолчанию copy)\n"
      "  --subs <copy|none>         субтитры из исходника (MKV; по умолчанию copy)\n"
      "\n"
      "Пакетный режим:\n"
      "  Все входы обрабатываются последовательно. Имя выхода — <stem>.<ext>,\n"
      "  по умолчанию MP4 в <папка входа>\\upscaled. Папка на входе разворачивается\n"
      "  в видеофайлы верхнего уровня. Ошибка одного файла не останавливает пакет.\n"
      "\n"
      "Профиль:\n"
      "  --profile <name>           fast|balanced|quality|quality-split\n"
      "  --list-profiles            показать профили и выйти\n"
      "\n"
      "Апскейл/энкодер:\n"
      "  --scale <1..8>             множитель (по умолчанию 2)\n"
      "  --preset <p1..p7|quality>  пресет NVENC\n"
      "  --tune <ull|ll|hq>         tuning info\n"
      "  --rc <cbr|cqp>             режим rate control\n"
      "  --qp <0..51>               QP для CQP\n"
      "  --bitrate <kbps>           битрейт для CBR\n"
      "  --gop <1..10000>           GOP length\n"
      "  --bframes <0..5>           число B-кадров\n"
      "  --ref <0..16>              ref frames / DPB (0 = auto)\n"
      "  --lookahead <0..31>        lookahead depth\n"
      "  --bref-mode <mode>         middle|disabled|auto\n"
      "  --encoders <1|2>           число NVENC-сессий (chunked GOP)\n"
      "  --split-enc <mode>         auto|auto_forced|forced_2|forced_3|disable\n"
      "  --bit-depth <8|10>         выход: NV12 или P010/Main10\n"
      "  --10bit                    алиас --bit-depth 10\n"
      "\n"
      "Диагностика:\n"
      "  --frames <N>               остановиться после N кадров (0 = все)\n"
      "  --gray                     серый кадр вместо compute (проверка тракта)\n"
      "  --no-audio                 не копировать аудио\n"
      "  --no-subs                  не копировать субтитры\n"
      "  --live                     режим live (флаг зарезервирован)\n"
      "  --dry-run                  показать итоговый конфиг и выйти\n"
      "  --version, -V              версия\n"
      "  --help, -h                 эта справка\n"
      "\n"
      "Коды возврата:\n"
      "  0 — успех\n"
      "  1 — ошибка выполнения (CUDA/TRT/NVENC/файлы)\n"
      "  2 — неверные аргументы\n"
      "\n"
      "Примечание: --profile quality включает B=5/lookahead, поэтому chunked dual\n"
      "принудительно отключается. Для двух движков используйте --profile quality-split.\n"
      "Выход .mkv/.mp4 упаковывается через FFmpeg (аудио/субтитры), .h265/.h264 — raw.\n"
      "С --onnx готовый engine подхватывается, иначе собирается под текущую GPU\n"
      "и разрешение и кэшируется рядом с ONNX; --rebuild-engine форсирует сборку.\n");
}

void PrintResolved(const CliOptions& opt) {
  const PipelineConfig& c = opt.cfg;
  std::printf("profile: %s\n", opt.profile.empty() ? "defaults" : opt.profile.c_str());
  std::printf("in: %s\n", c.in_path.empty() ? "(не задан)" : c.in_path.c_str());
  if (!c.engine_path.empty())
    std::printf("engine: %s\n", c.engine_path.c_str());
  if (!c.onnx_path.empty())
    std::printf("onnx: %s%s\n", c.onnx_path.c_str(),
                c.rebuild_engine ? " (пересборка)" : "");
  std::printf("out: %s\n", c.out_path.empty() ? "(не задан)" : c.out_path.c_str());
  std::printf("scale: %d\n", c.scale);
  std::printf("out-codec: %s\n", c.out_codec.c_str());
  std::printf("bit-depth: %d\n", c.bit_depth);
  std::printf("encoder: preset=%s tune=%s rc=%s qp=%d bitrate=%d gop=%d bf=%d ref=%d la=%d bref=%s split=%s sessions=%d\n",
              c.preset.c_str(), c.tune.c_str(), c.rc.c_str(), c.qp, c.bitrate_kbps,
              c.gop, c.bframes, c.refs, c.lookahead, c.bref_middle ? "middle" : "disabled",
              c.split_enc.c_str(), c.num_encoders);
  std::printf("frames: %d%s\ngray: %d  live: %d\n", c.max_frames,
              c.max_frames == 0 ? " (все)" : "", c.gray ? 1 : 0, c.live_drop ? 1 : 0);
  std::printf("audio: %s  subs: %s\n", c.copy_audio ? "copy" : "none",
              c.copy_subs ? "copy" : "none");
  if (c.out_codec == "h264" && c.bit_depth == 10)
    std::printf("warning: H.264 не поддерживает 10-bit, runtime переключится на 8-bit\n");
  if (c.num_encoders > 1 && (c.bframes > 0 || c.lookahead > 0 || c.split_enc != "disable"))
    std::printf("warning: B-кадры/lookahead/split требуют 1 сессии, runtime понизит sessions до 1\n");
}
