#pragma once
// Пакетная обработка: входы разворачиваются, выходы получают имя исходника.
#include "Cli.h"
#include <string>

struct Job {
  std::string in;
  std::string out;
};

// 0 — все файлы успешно; 1 — были ошибки; 2 — неверные входы/выходы.
int RunBatch(const CliOptions& opt);
