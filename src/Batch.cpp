// Пакетный режим: разворачивает папки в видеофайлы, строит имена выходов
// (<stem>.<ext>, по умолчанию .mp4) и гоняет Pipeline последовательно.
#include "Batch.h"
#include <windows.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::wstring Widen(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (n <= 0) return {};
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  w.resize((size_t)n - 1);
  return w;
}

std::string Narrow(const std::wstring& w) {
  if (w.empty()) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (n <= 0) return {};
  std::string s((size_t)n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
  s.resize((size_t)n - 1);
  return s;
}

std::string Lower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

std::string PathKey(const fs::path& p) {
  std::error_code ec;
  fs::path abs = fs::weakly_canonical(fs::absolute(p, ec), ec);
  if (ec) abs = p;
  std::wstring w = abs.wstring();
  if (!w.empty()) CharLowerBuffW(w.data(), (DWORD)w.size());
  return Narrow(w);
}

bool IsVideoFile(const fs::path& p) {
  static const char* kExts[] = {".mkv", ".mp4", ".m4v", ".mov", ".ts", ".m2ts",
                                ".webm", ".avi", ".wmv", ".mpg", ".mpeg", ".flv"};
  const std::string ext = Lower(p.extension().u8string());
  for (const char* e : kExts) {
    if (ext == e) return true;
  }
  return false;
}

bool ExpandInputs(const std::vector<std::string>& raw, std::vector<fs::path>& out,
                  std::string& error) {
  std::set<std::string> seen;
  auto add = [&](const fs::path& p) {
    const std::string key = PathKey(p);
    if (!seen.insert(key).second) return;
    std::error_code ec;
    fs::path abs = fs::absolute(p, ec);
    out.push_back(ec ? p : abs);
  };

  for (const std::string& item : raw) {
    if (item.empty()) continue;
    const fs::path p = fs::u8path(item);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
      std::vector<fs::path> found;
      for (const auto& entry : fs::directory_iterator(p, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        if (IsVideoFile(entry.path())) found.push_back(entry.path());
      }
      if (found.empty()) {
        error = "в папке нет видеофайлов: " + item;
        return false;
      }
      std::sort(found.begin(), found.end(), [](const fs::path& a, const fs::path& b) {
        return Lower(a.filename().u8string()) < Lower(b.filename().u8string());
      });
      for (const fs::path& f : found) add(f);
      continue;
    }
    if (!fs::is_regular_file(p, ec)) {
      error = "вход не найден: " + item;
      return false;
    }
    add(p);
  }
  if (out.empty()) {
    error = "нет входных файлов";
    return false;
  }
  return true;
}

std::string OutExt(const CliOptions& opt) {
  std::string ext = opt.out_ext.empty() ? "mp4" : Lower(opt.out_ext);
  if (!ext.empty() && ext[0] == '.') ext.erase(0, 1);
  return ext;
}

bool ResolveJobs(const CliOptions& opt, std::vector<Job>& jobs, std::string& error) {
  std::vector<fs::path> inputs;
  if (!ExpandInputs(opt.inputs, inputs, error)) return false;

  if (!opt.cfg.out_path.empty()) {
    if (inputs.size() != 1) {
      error = "один -o <файл> нельзя использовать для нескольких входов";
      return false;
    }
    Job job;
    job.in = inputs[0].u8string();
    job.out = fs::u8path(opt.cfg.out_path).u8string();
    if (PathKey(fs::u8path(job.in)) == PathKey(fs::u8path(job.out))) {
      error = "выходной файл совпадает с входным: " + job.out;
      return false;
    }
    jobs.push_back(std::move(job));
    return true;
  }

  const std::string ext = OutExt(opt);
  std::set<std::string> used;
  for (const fs::path& in : inputs) {
    fs::path dir;
    if (!opt.out_dir.empty()) {
      dir = fs::u8path(opt.out_dir);
    } else {
      dir = in.parent_path() / L"upscaled";
    }
    const fs::path out = dir / fs::u8path(in.stem().u8string() + "." + ext);
    if (PathKey(in) == PathKey(out)) {
      error = "выход совпадает с входом (" + out.u8string() +
              "); укажите другой --out-dir";
      return false;
    }
    if (!used.insert(PathKey(out)).second) {
      error = "два входа дают один выход: " + out.u8string() +
              "; переименуйте файлы или используйте разные папки";
      return false;
    }
    Job job;
    job.in = in.u8string();
    job.out = out.u8string();
    jobs.push_back(std::move(job));
  }

  // Выход одного файла не должен совпадать с входом другого: иначе первый
  // job перезапишет исходник до того, как второй начнёт его читать.
  std::set<std::string> input_keys;
  for (const fs::path& in : inputs) input_keys.insert(PathKey(in));
  for (const Job& job : jobs) {
    if (input_keys.count(PathKey(fs::u8path(job.out)))) {
      error = "выход " + job.out +
              " совпадает с одним из входов; укажите другой --out-dir";
      return false;
    }
  }
  return true;
}

} // namespace

int RunBatch(const CliOptions& opt) {
  // Сухой прогон без входов — показать дефолты/профиль, как раньше.
  if (opt.dry_run && opt.inputs.empty()) {
    PrintResolved(opt);
    return 0;
  }

  std::vector<Job> jobs;
  std::string error;
  if (!ResolveJobs(opt, jobs, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 2;
  }

  const bool batch = jobs.size() > 1;
  if (opt.dry_run) {
    for (size_t i = 0; i < jobs.size(); ++i) {
      if (batch) std::printf("=== файл %zu/%zu ===\n", i + 1, jobs.size());
      CliOptions one = opt;
      one.cfg.in_path = jobs[i].in;
      one.cfg.out_path = jobs[i].out;
      PrintResolved(one);
      if (batch && i + 1 < jobs.size()) std::printf("\n");
    }
    return 0;
  }

  size_t ok = 0, failed = 0;
  for (size_t i = 0; i < jobs.size(); ++i) {
    const Job& job = jobs[i];
    if (batch) {
      std::printf("[batch] %zu/%zu %s -> %s\n", i + 1, jobs.size(),
                  job.in.c_str(), job.out.c_str());
      std::fflush(stdout);
    }

    PipelineConfig cfg = opt.cfg;
    cfg.in_path = job.in;
    cfg.out_path = job.out;
    cfg.batch_index = (int)i + 1;
    cfg.batch_total = (int)jobs.size();

    {
      const fs::path out_dir = fs::u8path(job.out).parent_path();
      std::error_code ec;
      if (!out_dir.empty()) fs::create_directories(out_dir, ec);
      if (ec) {
        std::fprintf(stderr, "[batch] не удалось создать папку %s: %s\n",
                     out_dir.u8string().c_str(), ec.message().c_str());
        ++failed;
        continue;
      }
    }

    bool good = false;
    try {
      Pipeline pipeline;
      good = pipeline.run(cfg);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[batch] исключение: %s\n", e.what());
    } catch (...) {
      std::fprintf(stderr, "[batch] неизвестное исключение\n");
    }

    if (good) {
      ++ok;
      if (batch) {
        std::printf("[batch] готово %zu/%zu\n", i + 1, jobs.size());
        std::fflush(stdout);
      }
    } else {
      ++failed;
      if (batch) {
        std::fprintf(stderr, "[batch] ошибка %zu/%zu: %s\n", i + 1, jobs.size(),
                     job.in.c_str());
      } else {
        std::fprintf(stderr, "error: не удалось обработать %s\n", job.in.c_str());
      }
    }
  }

  if (batch) {
    std::printf("[batch] итог: ok=%zu failed=%zu\n", ok, failed);
    std::fflush(stdout);
  }
  return failed == 0 ? 0 : 1;
}
