#pragma once
// Приоритеты и affinity тредов под Windows. WDDM любит ровные потоки.
#include <windows.h>

inline void PinCurrentThread(int cpu_index, int priority) {
  DWORD_PTR mask = (DWORD_PTR)1 << (cpu_index % 64);
  SetThreadAffinityMask(GetCurrentThread(), mask);
  SetThreadPriority(GetCurrentThread(), priority);
}

// "Playback" через avrt.dll поднимает MMCSS-приоритет для decode/compute.
inline void EnableMmcssPlayback() {
  HMODULE avrt = LoadLibraryA("avrt.dll");
  if (!avrt) return;
  using Fn = HANDLE(WINAPI*)(LPCSTR, DWORD*);
  auto begin = (Fn)GetProcAddress(avrt, "AvSetMmThreadCharacteristicsA");
  if (begin) { DWORD task = 0; begin("Playback", &task); }
}
