#pragma once
// CLI-слой: разбор аргументов и профили. Pipeline о флагах не знает.
#include "Pipeline.h"
#include <string>
#include <vector>

enum class CliAction {
  Run,
  Help,
  Version,
  ListProfiles,
  Error,
};

struct CliOptions {
  PipelineConfig cfg;
  bool dry_run = false;
  std::string profile; // пусто = дефолты PipelineConfig
  // Пакетный режим: пути файлов/папок в порядке аргументов.
  // cfg.in_path/out_path заполняются на каждый файл в Batch.cpp.
  std::vector<std::string> inputs;
  std::string out_dir;   // --out-dir (иначе <папка входа>\upscaled)
  std::string out_ext;   // расширение для пакетного выхода (по умолчанию mp4)
};

// Возвращает Error и заполняет error при кривом наборе аргументов.
CliAction ParseCli(const std::vector<std::string>& args, CliOptions& opt, std::string& error);

void PrintHelp();
void PrintVersion();
void PrintProfiles();
void PrintResolved(const CliOptions& opt);
