// nisshoku — CLI-обертка вокруг GPU-пайплайна.
// Разбор аргументов и профили живут в Cli.cpp, Pipeline о флагах не знает.
#include "Cli.h"
#include "Batch.h"
#include "Logger.h"
#include <cstdio>
#include <string>
#include <vector>
#include <windows.h>
#include <dbghelp.h>

// Глобал, которого ждет SDK: NvDecoder/NvEncoder пишут логи через него.
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Временный диагност падений: печатает код + стек через DbgHelp.
static void PrintStack(CONTEXT* ctx) {
  HANDLE proc = GetCurrentProcess();
  SymInitialize(proc, nullptr, TRUE);
  void* frames[62];
  USHORT n = CaptureStackBackTrace(0, 62, frames, nullptr);
  char buf[sizeof(SYMBOL_INFO) + 256];
  SYMBOL_INFO* sym = (SYMBOL_INFO*)buf;
  sym->MaxNameLen = 255;
  sym->SizeOfStruct = sizeof(SYMBOL_INFO);
  for (USHORT i = 0; i < n; ++i) {
    DWORD64 addr = (DWORD64)frames[i];
    DWORD64 disp = 0;
    const char* name = "?";
    if (SymFromAddr(proc, addr, &disp, sym)) name = sym->Name;
    fprintf(stderr, "#%02u %p %s+0x%llx\n", i, frames[i], name, disp);
  }
  SymCleanup(proc);
}

static LONG WINAPI FailFastHandler(EXCEPTION_POINTERS* ep) {
  fprintf(stderr, "FATAL code=0x%08lx at %p\n",
          (unsigned long)ep->ExceptionRecord->ExceptionCode,
          ep->ExceptionRecord->ExceptionAddress);
  PrintStack(ep->ContextRecord);
  fflush(stderr);
  return EXCEPTION_EXECUTE_HANDLER;
}

static int RunWithArgs(const std::vector<std::string>& args) {
  // GUI/пайп: не копить вывод в блоки, иначе прогресс появляется только в конце.
  setvbuf(stdout, nullptr, _IOLBF, 8192);
  setvbuf(stderr, nullptr, _IONBF, 0);
  SetUnhandledExceptionFilter(FailFastHandler);

  CliOptions opt;
  std::string error;
  CliAction action = ParseCli(args, opt, error);
  switch (action) {
    case CliAction::Help:
      PrintHelp();
      return 0;
    case CliAction::Version:
      PrintVersion();
      return 0;
    case CliAction::ListProfiles:
      PrintProfiles();
      return 0;
    case CliAction::Error:
      fprintf(stderr, "error: %s\n\nПодсказка: nisshoku --help\n", error.c_str());
      return 2;
    case CliAction::Run:
      break;
  }

  return RunBatch(opt);
}

#ifdef _WIN32
// Wide-вход: пути с кириллицей/пробелами доходят без потерь.
// FFmpeg на Windows принимает UTF-8 (конвертит внутри), поэтому все
// приводим к UTF-8, а не к ANSI-кодовой странице процесса.
static std::string WideToUtf8(const wchar_t* w) {
  if (!w || !*w) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s((size_t)(n > 1 ? n - 1 : 0), '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
  return s;
}

int wmain(int argc, wchar_t** wargv) {
  // Русский текст в help/ошибках: консоль Windows по умолчанию cp866/cp1251.
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  std::vector<std::string> args;
  args.reserve((size_t)argc);
  for (int i = 0; i < argc; ++i) args.push_back(WideToUtf8(wargv[i]));
  return RunWithArgs(args);
}
#else
int main(int argc, char** argv) {
  return RunWithArgs(std::vector<std::string>(argv, argv + argc));
}
#endif
