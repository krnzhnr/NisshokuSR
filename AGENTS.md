# AGENTS.md — карта проекта и правила работы

> **Текущее имя проекта — NisshokuSR.** Ниже используются текущие имена
> артефактов (`nisshoku.exe`, `NisshokuSR`, `NISSHOKU_*`). При следующем
> переименовании пройди чек-лист в конце и обнови этот файл, README и CHANGELOG.
>
> Этот файл — проектная карта и база знаний. `README.md` — пользовательское
> поведение, CLI и **журнал всех встреченных багов**. `CHANGELOG.md` — история
> изменений. Если код и документ расходятся, прав код; документ обнови.

## 0. TL;DR для спешащего агента

* Это Windows-only offline-апскейлер видео: NVDEC -> CUDA pre-kernel ->
  TensorRT -> CUDA post-kernel -> NVENC -> FFmpeg mux. Не кроссплатформенный,
  не live-стример, не свой кодек и не свой инференс-движок.
* Основной язык — C++17. CUDA — только кернелы pre/post. Тяжёлое считают
  TensorRT, NVDEC/NVENC и FFmpeg через вендорские API.
* Один driver-primary CUDA-контекст на процесс, один управляющий тред,
  device-only steady state. Не добавляй контексты, потоки и DtoH «по месту».
* Не ломай: FP16 I/O TensorRT, BT.709 limited, P016 в старших битах, CFR/PTS,
  unlock кадра NVDEC ровно один раз, chunked/split-правила NVENC, пакетный
  режим и коды возврата.
* Перед правкой прочитай `README.md` (особенно «Грабли» и «Известные
  ограничения»), `CHANGELOG.md`, `version.txt`, `CMakeLists.txt` и затронутые
  файлы.
* Любая правка поведения = обновить README и CHANGELOG. Любой найденный баг =
  запись в README, даже если исправлен в этой же сессии.
* `build/`, `dist/`, `setup_output/`, `bench_results/`, `NisshokuSR.iss`,
  `PORTABLE.txt` и `dist/**/README.md` — генерируемые артефакты. Не правь их
  руками; правь источники и пересобирай.
* Не коммить и не пушить без явной просьбы. Репозиторий может быть без `.git`.

## Содержание

1. Что за проект.
2. Инварианты.
3. Быстрый старт.
4. Архитектура и поток кадра.
5. Карта репозитория.
6. CLI, профили и пакет.
7. Подсистемы: контракты и ловушки.
8. Форматы, цвет, CFR.
9. Сборка, portable, installer.
10. Тестирование.
11. Документация: README, CHANGELOG, AGENTS.
12. Рабочий процесс агента.
13. Известные баги и ограничения.
14. Troubleshooting.
15. Чек-лист переименования.
16. Definition of Done.

---

## 1. Что это за проект

### 1.1. Назначение

Офлайн-апскейл видео на NVIDIA GPU под Windows:

```
входной файл
  -> libavformat + BSF (AVCC -> Annex-B)
  -> NVDEC (NvDecoder, device frames)
  -> CUDA pre-kernel: NV12/P016 -> RGB planar FP16
  -> TensorRT enqueueV3 (FP16 engine)
  -> CUDA post-kernel: RGB planar FP16 -> NV12/P010, BT.709
  -> NVENC (NvEncoderCuda; chunked dual или split)
  -> временный MPEG-TS + ffmpeg.exe remux (аудио/субтитры/контейнер)
```

Цель — ноль `DtoH/HtoD` в steady-state: кадр рождается на GPU и умирает на GPU.
Хост читает только пакеты битстрима.

### 1.2. Что поддерживается

* Вход: контейнеры, которые читает FFmpeg (`.mkv`, `.mp4`, `.m4v`, `.mov`,
  `.ts`, `.m2ts`, `.webm`, `.avi`, `.wmv`, `.mpg`, `.mpeg`, `.flv`), плюс
  raw-потоки, если их понимает демуксер.
* Видеокодеки входа: H.264, HEVC, AV1, VP9 — через NVDEC; на практике
  проверены H.264 и HEVC.
* Выход: H.264 8-bit, HEVC 8/10-bit, AV1 8/10-bit (AV1 — только raw).
* Контейнеры выхода: MKV, MP4; raw `.h264`, `.h265`, `.av1`.
* GPU: RTX 20/30/40/50, бинарь под `sm_75;86;89;120`.
* Пакетный режим: несколько файлов/папка, последовательно.
* GUI: Tkinter-пульт для разработки, в portable не входит.

### 1.3. Что не поддерживается

* Linux/macOS.
* Live-стриминг: флаг `--live` зарезервирован, отдельного live-режима нет.
* AV1 в MKV/MP4: промежуточный MPEG-TS его не несёт; только raw `.av1`.
* Full-range и BT.601 источники: вход/выход считаются limited BT.709.
* Трёхтредовый конвейер decode/compute/encode: сейчас один управляющий тред.
* Свой H.264/HEVC-кодек, свой инференс-движок, свои контейнеры.
* Перенос TensorRT engine между GPU/версиями TRT: engine привязан.

---

## 2. Инварианты, которые нельзя ломать

Это самое важное. Если правка нарушает пункт — это не «оптимизация», а баг.

1. **Один CUDA-контекст на процесс.** `CudaContext` создаёт driver-primary
   через `cuCtxCreate`, выставляет его текущим и создаёт `comp_stream`.
   NVDEC, TensorRT и NVENC делят этот контекст. Не создавай отдельные контексты
   в обёртках, не делай `cuCtxPushCurrent/PopCurrent` вокруг SDK-вызовов:
   ломается `cuvidMapVideoFrame` и TensorRT. Проверка: короткий прогон +
   отсутствие `NV_ENC_ERR_MAP_FAILED`/`cuvidMap INVALID_*`.
2. **Только обычные blocking-стримы.** `comp_stream` создаётся без
   `cudaStreamNonBlocking` и без приоритета. Приоритетный non-blocking стрим
   ронял TRT-инференс на Blackwell (illegal access, порча всего контекста).
   Не «улучшай» это без отдельного расследования и теста на целевой GPU.
3. **TensorRT I/O строго FP16 с обеих сторон.** Загрузчик проверяет типы
   входного/выходного тензоров. Engine с FP32-I/O не подходит: конвейер
   выделяет FP16-буферы, чтение/запись уходит за границы, illegal access
   травит весь CUDA-контекст. При сборке из ONNX FP32-I/O принудительно
   переводится на FP16; при загрузке готового engine несовпадение — ошибка.
4. **Device-only steady state.** В горячем цикле нет `DtoH/HtoD`, кроме
   служебных копий драйвера. Отладочные дампы на хост — только временно и не
   в коммит. FFmpeg получает уже готовые пакеты битстрима.
5. **NVDEC-кадр разблокируется ровно один раз.** После `GetLockedFrame`
   поверхность живёт до `UnlockFrame`. В успешном пути анлок делается сразу
   после DtoD-копии в буфер энкодера; на ошибке/исключении — RAII. Двойной
   unlock и потерянный unlock — оба бага.
6. **Цвет: BT.709 limited (tv) снаружи, full RGB внутри.** Pre/post считают
   BT.709, chroma при выходе усредняется 2×2. P010/P016 хранят 10 бит в
   **старших** битах 16-битного слова. Не путать с LSB-упаковкой
   `yuv420p10le`. Full-range/BT.601 не поддерживаются.
7. **CFR и точный rational fps.** Частота берётся из исходника как
   `num/den` (например, `24000/1001`), без округления. В TS PTS/DTS
   сдвигаются, в MP4 задаётся `video_track_timescale` и edit list. B-кадры
   пишутся в decode-order; переставлять пакеты в display-order нельзя.
8. **Нет тихих дропов кадров.** Ошибка копии кадра, `EncodeFrame`, `flush`
   или дырка по PTS — это `failed` файла, а не «пропустить и продолжить».
   Пакетный режим продолжает со следующего файла. Битые пакеты декодера —
   отдельная политика: пропускаются, но считаются и печатаются в итоге.
9. **Правила NVENC-сессий.** Chunked dual (2 сессии) — только при `B=0` и
   `lookahead=0`; чанк = GOP, первый кадр чанка — IDR. При `B>0`/lookahead
   принудительно одна сессия. Split encoding — одна сессия, один кадр делится
   между движками; H.264 split не поддерживает; с `num_encoders>1` split
   оставляет одну сессию. Weighted prediction отключается при split и B-кадрах.
10. **Кэш engine валиден только по ключу.** Путь/размер/mtime ONNX, GPU,
    версия TRT, входное разрешение. Готовый `<stem>.engine` рядом с ONNX
    подхватывается первым. `--rebuild-engine` требует `--onnx` и в пакете
    срабатывает один раз на ключ «ONNX + разрешение + GPU».
11. **Муксинг только через временный MPEG-TS + ffmpeg.exe.** Видео пишется
    в `<out>.tmp.ts`, затем remux с аудио/субтитрами исходника. AV1 в MKV/MP4
    запрещён. Raw-выход годится только для профилей без B-кадров.
12. **Пути — UTF-8, файловые API — wide.** Внутри `std::string` — UTF-8.
    Для `fopen`/CRT используй `_wfopen`/wide-путь. ONNX-парсер TensorRT
    принимает narrow-путь в ACP: кириллица может не пройти, нужен латинский
    путь. Не возвращай ANSI-пути в остальной код.
13. **Пакетный режим.** Выход `<stem>.<ext>`, по умолчанию
    `<папка входа>\upscaled`, расширение `.mp4`. Коллизии выходов (включая
    регистр кириллицы) и совпадение выхода с любым входом проверяются до
    запуска. Ошибка одного файла не останавливает остальные.
14. **Генерируемые артефакты не редактируются вручную.** `build/`, `dist/`,
    `setup_output/`, `bench_results/`, `NisshokuSR.iss`, `PORTABLE.txt`,
    `dist/**/README.md` — результат сборки. Источник правды — `src/`,
    `scripts/`, `build_release.py`, корневой `README.md`.
15. **Документация — часть Definition of Done.** README и CHANGELOG
    обновляются вместе с кодом, а не «потом». Подробности — раздел 11.

---

## 3. Быстрый старт

### 3.1. Окружение

Нужны user-level переменные (после `setx` открыть новый терминал):

```powershell
NVCODEC_SDK_DIR = <корень Video Codec SDK 13.1+>
TRT_ROOT        = <корень TensorRT 10.8, ветка cuda-12.x>
FFMPEG_ROOT     = <корень FFmpeg shared 8.1 BtbN lgpl-shared>
CUDA_PATH       = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
```

Требования:

* Windows 11 x64, свежий драйвер NVIDIA, HAGS ON, Game Bar OFF.
* CUDA Toolkit 12.8/12.9 (12.x, **не** 13.x — под него собран TRT 10.8).
* TensorRT 10.8 (cuda-12.x), Video Codec SDK 13.1+, FFmpeg shared 8.1.
* MSVC v143 (VS2022). nvcc 12.8 падает на компиляторе VS18/2026
  (`cudafe` крэшится); `-allow-unsupported-compiler` не спасает.
* CMake/Ninja берутся из `third_party` через `scripts/build_windows.ps1`.

### 3.2. Сборка

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
# результат: build\nisshoku.exe
```

Скрипт сам подхватывает VS2022 `cl.exe`, portable CMake/Ninja и CUDA `bin`.
Не запускай «голый» `cmake` из другого окружения: получишь другой toolset или
ошибку CUDA props.

### 3.3. Запуск

> В исходном репозитории `testdata/` нет: примеры ниже предполагают, что ты
> положил свои файлы в `testdata/` (папка в `.gitignore`, в коммит не попадёт).
> Для чистого клона подставь свои пути/engine/ONNX.

```powershell
# Справка/профили/сухой прогон
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --help
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --list-profiles
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --dry-run --profile quality-split

# Короткий прогон 8-bit H.264
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i testdata\cut.mkv -e testdata\animejanai_fp16.engine `
  -o "$env:TEMP\out.mkv" --frames 10

# 10-bit HEVC, B=5, split
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i testdata\bench_src.mkv -e testdata\animejanai_fp16.engine `
  -o "$env:TEMP\out10.mkv" --profile quality-split --frames 20
```

`scripts/run.ps1` добавляет в `PATH` `TRT\lib`, `TRT\bin`, `FFmpeg\bin`,
`CUDA\bin`. Без этого exe падает с `0xc0000135`: TRT-DLL лежат в `lib`, а не
в `bin`.

### 3.4. Portable и installer

```powershell
run_build_release.bat          # build_release.py: build + dist + zip + setup
```

Результат:

```text
dist\NisshokuSR\                             — portable-папка
dist\NisshokuSR_v<version>_portable.zip      — ZIP
setup_output\NisshokuSR_v<version>_Setup.exe — installer (если найден ISCC)
```

`NISSHOKU_SKIP_BUILDER=1` исключает `nvinfer_builder_resource_10.dll` (~2 ГБ):
portable сможет работать только с готовыми engine, сборка из ONNX отключена.
`NISSHOKU_THIRD_PARTY` переопределяет путь к `third_party`.

---

## 4. Архитектура и поток кадра

### 4.1. Схема

```text
main (wmain, UTF-8 args)
  -> Cli::ParseCli + профили
  -> Batch::RunBatch (разворачивание входов, имена, коллизии)
       -> Pipeline::run (один файл)
            Demuxer (libavformat + BSF)
            CudaContext (primary + comp_stream)
            TrtUpscaler (engine/cache/ONNX)
            NvDecWrapper (NvDecoder)
            NvEncWrapper (NvEncoderCuda, 1-2 сессии)
            Muxer (tmp TS + ffmpeg.exe)
```

### 4.2. Путь одного кадра

1. `Demuxer::readNext` отдаёт Annex-B пакет (после BSF, если контейнер AVCC).
2. `NvDecWrapper::decodePacket` кормит NVDEC; `GetLockedFrame` возвращает
   device-указатель NV12/P016. Кадр **locked** до `unlock`.
3. `Pipeline` зануляет slack входного TRT-буфера.
4. Pre-kernel конвертирует NV12/P016 -> RGB planar FP16 `[1,3,H,W]`.
5. `TrtUpscaler::inferAsync` вызывает `enqueueV3` в `comp_stream`.
6. Post-kernel конвертирует RGB FP16 -> NV12/P010 с BT.709 и chroma 2×2.
7. `cudaStreamSynchronize(comp_stream)` — текущая точка сериализации.
8. `NvEncWrapper::encodeFromDevice` делает DtoD-копию в буфер энкодера,
   кодирует, складывает пакеты в reorder-буфер по PTS.
9. NVDEC-кадр разблокируется сразу после DtoD-копии.
10. `Muxer::write` пишет пакеты в временный MPEG-TS.
11. В конце `flush` энкодера/декодера, `Muxer::finish` через `ffmpeg.exe`,
    удаление временного TS (если не задан `NISSHOKU_KEEP_TS=1`).

### 4.3. Потоки и синхронизация

* Один управляющий тред. `PinCurrentThread(2, HIGHEST)` и MMCSS
  «Playback» — текущая настройка, не гарантия.
* Один `comp_stream` для pre-kernel, TRT и post-kernel.
* `cudaStreamSynchronize` на кадр — осознанная точка сериализации. Её
  удаление само по себе ничего не даёт: `cudaMemcpy2D` в `NvEncWrapper` идёт
  в legacy default stream и всё равно ждёт `comp_stream`. Настоящий выигрыш —
  только трёхтредовый конвейер с events и отдельными буферами.
* NVDEC/NVENC/TensorRT используют внутренние потоки, но вызываются с
  primary-контекстом; не добавляй `cudaSetDevice`/`cuCtxPushCurrent`.

### 4.4. Владение и lifetime

* `CudaContext` владеет primary-контекстом и `comp_stream`; обёртки контекст
  не создают и не разрушают.
* `TrtUpscaler` владеет runtime/engine/execution context; `setTensorAddress`
  вызывается при смене указателей.
* `NvDecWrapper` владеет `NvDecoder`; `DecodedFrame::dev_ptr` валиден только
  до `unlock`.
* `NvEncWrapper` владеет 1–2 сессиями `NvEncoderCuda`; входные буферы —
  внутренние буферы SDK, DtoD-копия синхронная.
* `Pipeline` владеет device-буферами `trt_in`, `trt_out`, `enc_nv12`;
  освобождает их после teardown обёрток.
* `Muxer` владеет временным TS и `AVFormatContext`; при ошибке сохраняет
  temp-файл для разбора.
* Порядок teardown: `enc.shutdown()` -> `trt.shutdown()` -> `dec.shutdown()`
  -> `cu.shutdown()`. Иначе TRT/NVENC сыпят «invalid device context».

---

## 5. Карта репозитория

| Путь | Назначение |
|---|---|
| `src/main.cpp` | `wmain`, UTF-8-аргументы, `simplelogger` для SDK, crash-стек, коды возврата. |
| `src/Cli.h/.cpp` | Разбор CLI, профили, `--help/--version/--list-profiles/--dry-run`. |
| `src/Batch.h/.cpp` | Разворачивание входов, сортировка, имена выходов, коллизии, цикл по файлам. |
| `src/Pipeline.h/.cpp` | Конфиг одного файла, выделение device-буферов, цикл кадров, прогресс, teardown. |
| `src/CudaContext.h/.cpp` | Primary CUDA-контекст + `comp_stream`; init/shutdown. |
| `src/Demuxer.h/.cpp` | libavformat, выбор видео-потока, BSF AVCC->Annex-B, fps rational, start_time. |
| `src/NvDecWrapper.h/.cpp` | `NvDecoder`, `GetLockedFrame`/`UnlockFrame`, счётчик ошибок декодера. |
| `kernels/pre_post.h/.cu` | CUDA-кернелы NV12/P016<->RGB FP16, BT.709, chroma 2×2, fill. |
| `src/TrtUpscaler.h/.cpp` | Выбор/сборка engine, кэш, ONNX, FP16 I/O, slack, `inferAsync`. |
| `src/NvEncWrapper.h/.cpp` | NVENC: конфиг, VUI, chunked dual/split, reorder, ошибки, flush. |
| `src/Muxer.h/.cpp` | Временный MPEG-TS, ffmpeg remux, PTS/DTS, аудио/субтитры, trim. |
| `src/WinThread.h/.cpp` | `PinCurrentThread`, MMCSS; `.cpp` — пустышка для линка. |
| `CMakeLists.txt` | C++17/CUDA, arch `75;86;89;120`, зависимости, `/O2 /GL`. |
| `scripts/build_windows.ps1` | Сборка Ninja + portable CMake + VS2022 + CUDA. |
| `scripts/run.ps1` | PATH для TRT/FFmpeg/CUDA и запуск `build\nisshoku.exe`. |
| `scripts/bench.ps1` | Замер: per-core CPU + GPU SM/MEM/ENC/DEC/JPG/OFA, логи и CSV. |
| `build_release.py` | Portable, zip, Inno Setup, DLL, README, версия. |
| `run_build_release.bat` | Обёртка запуска `build_release.py`. |
| `NisshokuSR.iss` | **Генерируется** `build_release.py`; не править вручную. |
| `version.txt` | Версия для CMake и release. |
| `tools/export_onnx.py` | Экспорт bilinear/tiny ONNX; `PYTHONUTF8=1` на cp1251-консоли. |
| `tools/build_engine.bat` | `trtexec` с FP16 I/O; движок непереносим. |
| `tools/gui.py` | Tkinter-пульт, парсит `[progress]`; в portable не входит. |
| `tools/inspect_dumps.py` | Разбор отладочных дампов. |
| `configs/pipeline.json` | **Не читается рантаймом**, только копируется в portable. |
| `testdata/` | Локальные фикстуры/движки; в исходном репозитории отсутствует и исключена через `.gitignore`. |
| `build/` | Генерируемая сборка; источник exe для `run.ps1`. |
| `dist/`, `setup_output/` | Генерируемые portable/installer; могут быть старее `build/`. |
| `bench_results/` | Результаты `bench.ps1`; не править. |
| `README.md` | Пользовательское поведение + **журнал багов** + ограничения. |
| `docs/CLI.md` | Полный CLI-справочник: флаги, дефолты, профили, форматы, примеры. |
| `CHANGELOG.md` | История изменений по версиям. |
| `AGENTS.md` | Этот файл: карта, инварианты, процессы. |

### 5.1. Быстрый указатель по задачам

| Задача | Куда смотреть |
|---|---|
| Цвет уехал / BT.709 / P010 | `kernels/pre_post.cu`, `NvEncWrapper` VUI, `Muxer` color tags |
| CFR / VFR / PTS / аудио | `Demuxer` fps/start_time, `Muxer` setts/timescale/trim |
| Engine/ONNX/кэш | `TrtUpscaler`, `tools/export_onnx.py`, `tools/build_engine.bat` |
| NVENC split/chunked/B-кадры | `NvEncWrapper` |
| Пакет/имена/коллизии | `Batch`, `Cli` |
| Portable/installer/версия | `build_release.py`, `version.txt`, `NisshokuSR.iss` |
| GUI/прогресс | `tools/gui.py`, формат `[progress]` в `Pipeline` |
| Сборка/архитектуры | `CMakeLists.txt`, `scripts/build_windows.ps1` |
| Замеры | `scripts/bench.ps1`, `bench_results/` |

### 5.2. Про `testdata`

* В исходном репозитории папки `testdata/` нет: она исключена через
  `.gitignore`. Это сделано специально — там тестовые клипы, модели и engine.
* Для локальных прогонов положи свои файлы в `testdata/`; в коммит они не
  попадут. Не добавляй чужие видео/веса в git.
* Если положишь `animejanai_fp16.engine` — он привязан к GPU/TRT (текущий
  вариант был под RTX 5070 Ti / TRT 10.8 / 1080p). На другой карте нужен
  `--onnx` или свой engine.
* `model*.onnx` — тестовые/рабочие модели; в репозиторий не входят.
* Дампы `dump_*.nv12`, `dump_*.f16`, `trt_out_dump.json` — отладочные,
  не эталонные файлы; не коммитить.
* Большие `combat_*.mkv`, `bench_*.h265`, `cut*.h265` — локальные тестовые
  прогоны; не коммитить.

---

## 6. CLI, профили и пакетный режим

### 6.1. Ключевые флаги

| Флаг | Смысл |
|---|---|
| `-i, --in <path>` | Вход; можно повторять; позиционные пути тоже входы. |
| `-e, --engine <path>` | Готовый TensorRT engine. |
| `--onnx <path>` | ONNX: готовый engine подхватится, иначе соберётся. |
| `--rebuild-engine` | Игнорировать engine/кэш; требует `--onnx`. |
| `-o, --out <path>` | Файл для одного входа; для нескольких — папка/`--out-dir`. |
| `--out-dir <dir>` | Папка пакетного выхода. |
| `--out-ext <ext>` | `mp4|mkv|h264|h265|av1|raw`, по умолчанию `mp4`. |
| `--codec-out <c>` | `h264|hevc|h265|av1`; AV1 — только raw. |
| `--audio copy|none` | Копировать аудио. |
| `--subs copy|none` | Копировать субтитры (MKV). |
| `--profile` | Профиль; явные флаги перекрывают его. |
| `--scale`, `--preset`, `--tune`, `--rc`, `--qp`, `--bitrate`, `--gop` | Настройки энкодера. |
| `--bframes`, `--ref`, `--lookahead`, `--bref-mode` | B/DPB/lookahead. |
| `--encoders <1|2>` | Число NVENC-сессий (chunked GOP). |
| `--split-enc` | `auto|auto_forced|forced_2|forced_3|disable`. |
| `--bit-depth`, `--10bit` | NV12 или P010/Main10. |
| `--frames N` | Диагностический стоп после N кадров. |
| `--gray` | Серый кадр вместо compute. |
| `--dry-run` | Показать конфиг и выйти. |

`--scale` не масштабирует вход: фактический коэффициент берётся из engine,
несовпадение с флагом печатается. `--out-ext av1`/`--codec-out av1` дают raw
поток; AV1 в MKV/MP4 запрещён на уровне `Muxer`.

`-o <файл>` тоже проверяется: `.h265`/`.hevc` требуют `--codec-out hevc`,
`.h264` — `h264`, `.av1` — `av1`; `.raw` или файл без расширения — сырой
поток с кодеком из `--codec-out`. Несовпадение — код 2 до запуска.
Raw-выход при `bframes > 0` запрещён (код 2): используйте `.mkv`/`.mp4` или
`--bframes 0`.

### 6.2. Профили

| Профиль | Кодек | Preset/tune | RC | B | ref/LA | Глубина | Сессии |
|---|---|---|---|---|---|---|---|
| `fast` | HEVC | P1/HQ | CQP18 | 0 | 0/0 | 8-bit | 2 (chunked) |
| `balanced` | HEVC | P7/HQ | CQP18 | 0 | 0/0 | 10-bit | 2 (chunked) |
| `quality` | HEVC | P7/HQ | CQP18 | 5 | 8/16 | 10-bit | 1 |
| `quality-split` | HEVC | P7/HQ | CQP18 | 5 | 8/16 | 10-bit | 1 + split forced_2 |

Все профили ставят GOP 250, `bref-middle` для `quality*`. Явные флаги
перекрывают профиль **независимо от порядка** аргументов (CLI делает два
прохода: сначала профиль, потом явные опции). Старые имена `svfi*` и флаг
`--svfi` удалены.

### 6.3. Пакет

* Папка разворачивается в видеофайлы **верхнего уровня** и сортируется по имени.
* Выход: `<stem>.<ext>`, по умолчанию `<папка входа>\upscaled\<stem>.mp4`.
* Коллизии выходов (включая регистр кириллицы) и совпадение выхода с любым
  входом — ошибка **до** запуска.
* Ошибка одного файла не останавливает остальные; итог
  `[batch] итог: ok=N failed=M`; код возврата 1, если есть failed.
* Один `-o <файл>` допустим только для одного входа. Для нескольких —
  `--out-dir`/`--out-ext` или папка.

### 6.4. Коды возврата

* `0` — успех.
* `1` — ошибка выполнения (хотя бы один файл пакета failed).
* `2` — неверные аргументы/входы/выходы.

### 6.5. Контракт прогресса

CLI печатает раз в ~2 секунды:

```text
[progress] frames=N total=M pct=P fps=F speed=S elapsed=E eta=T
```

В пакете добавляются `file=K files=L`. GUI парсит именно эту строку; не меняй
формат без обновления `tools/gui.py`. Финальная строка:

```text
done, frames=N, S s, F fps -> <path>
failed, frames=N, S s, F fps -> <path>
```

`scripts/bench.ps1` ищет `done, frames=...`; при failed бенчмарк не должен
считать прогон успешным.

### 6.6. Добавление флага, профиля, кодека

* **Новый CLI-флаг:** `ParseOption`/`ParseCli` в `Cli.cpp`, затем `PrintHelp`.
  Если флаг влияет на пайплайн — поле в `PipelineConfig` и использование в
  `Pipeline`/`NvEncWrapper`. Обнови README и CHANGELOG; если флаг виден в GUI —
  добавь в `tools/gui.py`. Явные флаги должны перекрывать профиль.
* **Новый профиль:** `ApplyProfile`, `PrintProfiles` в `Cli.cpp`, таблица в
  README и `PROFILES` в `tools/gui.py` (GUI дублирует профили вручную —
  синхронизируй). Проверь, что явные флаги по-прежнему перекрывают профиль.
* **Новый кодек/формат:** `NvEncWrapper` (GUID, VUI, bit depth), `Muxer`
  (codec_id, MPEG-TS/контейнер), `Pipeline` (глубина, пост-процесс), имя
  расширения. Проверь 8/10-bit, container и raw. AV1 в MKV/MP4 не обещай.
* **Новая модель:** ONNX 1 вход/1 выход, NCHW, 3 канала, FP16 I/O. Проверь
  сборку engine, кэш, другое разрешение/GPU, обнови README.

---

## 7. Подсистемы: контракты и ловушки

### 7.1. `CudaContext`

* `init()`: `cuInit`, `cuDeviceGet`, `cuCtxCreate(BLOCKING_SYNC)`,
  `cuCtxSetCurrent`, `cudaStreamCreate(comp_stream)`.
* Один контекст на процесс; обёртки берут текущий через `cuCtxGetCurrent`.
* `shutdown()`: вернуть контекст, уничтожить stream, `cuCtxDestroy`.
* Не добавлять сюда пулы потоков/events «на будущее»: они создают лишние
  точки отказа и выглядят как рабочая архитектура. Если нужен новый
  стрим — добавляй вместе с реальным использованием и тестом.

### 7.2. `Demuxer`

* Открывает через `avformat_open_input` с `buffer_size=4194304`.
* Выбирает первый видео-поток.
* BSF:
  * H.264 -> `h264_mp4toannexb`;
  * HEVC -> `hevc_mp4toannexb`.
  * Контракт BSF: при успехе фильтр забирает владение пакетом; `unref` после
    успешного `send_packet` делать нельзя. При ошибке пакет не тронут.
* `start_us` — минимум по всем потокам; `video_start_us` — start_time видео.
  Это нужно для `-itsoffset` при remux.
* fps — `avg_frame_rate`, иначе `r_frame_rate`, иначе 30/1. Не округлять:
  иначе MP4 получает VFR.
* `estimatedFrames` — `nb_frames`, иначе duration*fps; это только прогресс.

### 7.3. `NvDecWrapper`

* `NvDecoder` создаётся на primary-контексте, device frames, без pitch.
* `decodePacket` ловит `NVDECException`/`std::exception`: пакет пропускается,
  счётчик `decode_errors` растёт. Это политика для повреждённого входа.
* `flush` тоже ловит исключения, но возвращает `false`: потеря хвоста —
  фатальная ошибка файла.
* `unlock(dev_ptr)` вызывает `UnlockFrame(&p, 1)`; ровно один раз на кадр.
* `DecodedFrame::dev_ptr` валиден только до unlock. Не сохранять его в
  долгоживущие структуры.

### 7.4. Pre/post kernels

* `launch_nv12_to_rgb_planar_fp16`, `launch_p016_to_rgb_planar_fp16`:
  NV12/P016 -> RGB planar FP16 `[1,3,H,W]`.
* `launch_rgb_planar_fp16_to_nv12`, `launch_rgb_planar_fp16_to_p010`:
  обратно, BT.709, chroma 2×2.
* `launch_fill_p010` — диагностический серый кадр.
* P016/P010: 10 бит в старших битах; при чтении/записи не путать с LSB.
* Full-range/BT.601 не реализованы; не добавляй «универсальную» матрицу без
  тестов на реальных источниках.

### 7.5. `TrtUpscaler`

Порядок выбора engine:

1. `--rebuild-engine` + `--onnx`: пропустить готовый engine и кэш.
2. Явный `-e/--engine`: если файл есть — загрузить; если не подошёл и есть
   ONNX — пересобрать.
3. `<stem>.engine` рядом с ONNX.
4. Кэш `<stem>_<W>x<H>_<gpu>_fp16_trt<ver>.engine` рядом с ONNX.
5. Fallback-кэш `%LOCALAPPDATA%\NisshokuSR\engines`.
6. Сборка из ONNX, сохранение кэша, загрузка в память.

Детали:

* Метаданные кэша: путь/размер/mtime ONNX, input WxH, GPU, TRT, `io=fp16`.
* `--rebuild-engine` без `--onnx` — ошибка (код 2 на уровне CLI).
* В пакете rebuild дедуплицируется по ключу ONNX+WxH+GPU.
* ONNX: ровно 1 вход/1 выход, NCHW, 3 канала, батч 1 или динамический.
* Статический вход должен совпадать с видео; динамический получает профиль
  `[1,3,H,W]`.
* FP32 I/O принудительно переводится в FP16; готовый engine с не-FP16 I/O
  отвергается.
* `kIoSlack = 8 MiB` на вход и выход. Зануляется только slack-хвост:
  сам входной тензор пишет pre-kernel, выходной — engine. Полный memset
  вернёт лишние ~59 МиБ/кадр и не нужен.
* `inDataBytes()/outDataBytes()` — размеры тензоров без slack;
  `inBytes()/outBytes()` — с slack. Не путать.
* `inferAsync` только ставит работу в stream; синхронизация — на вызывающем.
* `setTensorAddress` вызывается при смене указателей; адреса хранятся в
  `Impl`, не в `static`.

### 7.6. `NvEncWrapper`

* 1–2 сессии `NvEncoderCuda` на primary-контексте.
* `num_encoders=2` — chunked GOP: чётные GOP на enc0, нечётные на enc1;
  чанк = GOP; первый кадр чанка — принудительный IDR.
* Chunked dual несовместим с B/lookahead: при `bframes>0 || lookahead>0`
  принудительно 1 сессия.
* `split_enc` — одна сессия, кадр делится между движками; H.264 не
  поддерживает; при split `num_encoders` принудительно 1; weighted prediction
  выключается.
* B>0: пакеты идут в decode-order, не переставлять. B=0: reorder-буфер по
  глобальному PTS, `drainContiguous`.
* `picParams.inputTimeStamp = pts+1`; 0 зарезервирован.
* DtoD-копия luma/chroma в буфер энкодера; при сбое — один повтор, затем
  fatal. Копия идемпотентна.
* Ошибка `EncodeFrame`/`flush` — fatal для файла; не возвращать `true`.
* `drainContiguous` при дырке >64 кадров — fatal, не «skip gap».
* `flush` возвращает `false`, если хотя бы одна сессия не смогла отдать хвост.
* VUI: BT.709, limited/tv, timing info.

### 7.7. `Muxer`

* Пишет видео в `<out>.tmp.ts` (MPEG-TS) через libavformat.
* `finish` запускает `ffmpeg.exe`:
  `-copyts`, `-i tmp.ts`, `-itsoffset` для базы исходника, `-i source`,
  `-map 0:v:0`, аудио/субтитры из source, `-c copy`, `-map_metadata 1`.
* MP4: `-map_chapters -1`, `-video_track_timescale fps_num`,
  `-use_editlist 1`, `-movflags +faststart`.
* MKV: `-f matroska`, главы/метаданные сохраняются.
* `-bsf:v setts=pts=...:dts=...` сдвигает видео назад после TS.
* `trim_to_video` (`--frames`) добавляет `-t`; для Opus это известный баг.
* `NISSHOKU_KEEP_TS=1` оставляет временный TS; иначе удаляется.
* AV1 в MKV/MP4 — ошибка на `open`; raw AV1 — отдельный путь.
* При ошибке ffmpeg temp TS сохраняется для разбора; лог рядом.

### 7.8. `Pipeline`

* Открывает demux, CudaContext, TRT, decoder, encoder, muxer.
* Выделяет `trt_in`, `trt_out`, `enc_nv12` через `cuMemAlloc`.
* `processFrame`: pre -> TRT -> post -> sync -> encode -> unlock -> mux.
* `--frames` проверяется до обработки; необработанные кадры из текущего
  пакета разблокируются `unlock_from`.
* `--gray` пропускает compute/TRT, заливает серый кадр.
* Прогресс: `estimatedFrames`, `--frames` ограничивает total.
* `flush` энкодера и декодера проверяются; ошибка = failed.
* Финальная строка `done`/`failed`; возврат `!fatal`.
* Teardown: enc -> trt -> dec -> mux finish -> device free -> CUDA shutdown.
* Не добавляй сюда «второй Pipeline»/треды без пересмотра владения буферами.

### 7.9. Batch и Cli

* `Cli` не знает о Pipeline-реализации; заполняет `PipelineConfig`.
* `Batch` разворачивает входы, строит `Job{in,out}`, создаёт `Pipeline` на
  каждый файл.
* `PathKey`: absolute + weakly_canonical + lowercase wide (`CharLowerBuffW`),
  чтобы ловить коллизии кириллицы/регистра.
* Проверка «выход = вход другого job» обязательна.
* Ошибка одного файла не прерывает пакет.
* `--dry-run` без входов печатает дефолты; с входами — резолв по каждому.

### 7.10. GUI и tools

* `tools/gui.py` — dev-пульт: список входов, ONNX, engine, out-dir,
  контейнер, профиль, прогресс. Запускает `build\nisshoku.exe` или
  `dist\NisshokuSR\nisshoku.exe`.
* GUI не входит в portable; не обещай его пользователям portable.
* `tools/export_onnx.py`: `--arch bilinear|tiny`, `--dtype fp16|fp32`,
  `--ckpt` только для tiny. `PYTHONUTF8=1` на cp1251-консоли.
* `tools/build_engine.bat`: `trtexec --fp16 --stronglyTyped` + FP16 I/O;
  без этого engine не подойдёт.
* `tools/inspect_dumps.py` — разбор дампов, не production.

---

## 8. Форматы, цвет, CFR

### 8.1. Форматы

| Этап | Формат |
|---|---|
| NVDEC output 8-bit | NV12 (Y + interleaved UV) |
| NVDEC output 10-bit | P016 (10 бит в MSB 16-битного слова) |
| TRT input/output | RGB planar FP16, NCHW `[1,3,H,W]` |
| NVENC input 8-bit | NV12 |
| NVENC input 10-bit | P010 (10 бит в MSB) |
| Временный контейнер | MPEG-TS |
| Выход | MKV/MP4 или raw H.264/HEVC/AV1 |

### 8.2. Цвет

* Вход и выход: BT.709 limited (tv).
* Внутри: full RGB.
* Pre: limited YUV -> full RGB по BT.709.
* Post: full RGB -> limited YUV по BT.709; chroma 2×2 усредняется.
* VUI NVENC: primaries/transfer/matrix BT.709, full_range=0.
* Full-range/BT.601 не поддерживаются. Не добавляй эвристику по метаданным
  без тестов и обновления README.

### 8.3. CFR/PTS

* fps — точный rational из исходника.
* TS: PTS/DTS сдвигаются, чтобы PTS >= DTS; при remux сдвиг вычитается.
* MP4: `video_track_timescale = fps_num`, edit list для B-кадров.
* B>0: пакеты в decode-order; не сортировать.
* B=0: reorder-буфер по PTS; дырка — fatal.
* Raw-выход: только B=0; для B>0 raw-демуксер теряет порядок.

---

## 9. Сборка, portable, installer

### 9.1. CMake

* C++17 и CUDA C++17, Release x64.
* `CMAKE_CUDA_ARCHITECTURES "75;86;89;120"` — RTX 20/30/40/50.
* `prepost_kernels` — отдельная static-либа, separable compilation.
* `nvcodec_utils` — NvDecoder/NvEncoder из SDK, без правок.
* Линковка: TRT `nvinfer_10`, `nvinfer_plugin_10`, `nvonnxparser_10`;
  FFmpeg `avformat/avcodec/avutil/swresample`; CUDA driver/runtime/NPP.
* Флаги: `/O2 /GL`, CUDA `--use_fast_math -O3`, `/LTCG`.

### 9.2. Portable

`build_release.py`:

* читает `version.txt`;
* запускает `scripts/build_windows.ps1`;
* копирует exe, `cudart64_12.dll`, TRT DLL, FFmpeg DLL + `ffmpeg.exe`,
  VC runtime, README, `configs/pipeline.json`;
* копирует `engines/animejanai_fp16.engine` только если он есть локально
  в `testdata`; иначе portable собирается без готового engine (используйте
  `--onnx`);
* пишет `PORTABLE.txt`, `nisshoku.cmd`;
* собирает zip;
* генерирует `NisshokuSR.iss` и запускает ISCC, если найден.

`NISSHOKU_SKIP_BUILDER=1` — без builder resource (~2 ГБ), только готовые engine.
`NISSHOKU_THIRD_PARTY` — другой путь к зависимостям.

### 9.3. Installer

* Inno Setup 6; `AppId=krnzhnr.nisshokusr`; `PrivilegesRequired=lowest`.
* Per-user, без администратора; по умолчанию `{autopf}` →
  `%LOCALAPPDATA%\Programs\NisshokuSR`.
* Ярлык «NisshokuSR CLI» запускает `cmd /k nisshoku.cmd`: печатает help и
  оставляет консоль открытой. `PATH` не меняется.
* Копирует builder resource по умолчанию; без `NISSHOKU_SKIP_BUILDER=1`
  сборка engine из ONNX после установки работает.
* Модель и готовый engine не входят; GUI не входит; установщик и exe не
  подписаны (SmartScreen может предупредить).
* Проверяет наличие `nvcuvid.dll`/`nvEncodeAPI64.dll` и предупреждает.
* `NisshokuSR.iss` генерируется; не править вручную.

### 9.4. Версия

* `version.txt` — единственный источник версии для CMake и release.
* `--version` печатает `NISSHOKU_VERSION`.
* При выпуске: bump `version.txt`, обновить CHANGELOG, пересобрать release.

---

## 10. Тестирование и проверка

### 10.1. Что есть

Фреймворка юнит-тестов нет. Проверка — интеграционная: собрать, прогнать
короткие файлы, проверить кадры/CFR/цвет/пакет/portable. `scripts/bench.ps1` —
замер, не тест.

### 10.2. Минимальный smoke после правок

> В чистом клоне `testdata/` нет; примеры ниже предполагают локальные файлы.
> Подставь свои входы, engine/ONNX и пути.

```powershell
# 1. Сборка
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1

# 2. Help/профили
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --help
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --list-profiles

# 3. 10 кадров 8-bit H.264
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i testdata\cut.mkv -e testdata\animejanai_fp16.engine `
  -o "$env:TEMP\smoke.mkv" --frames 10
# ожидаем: exit 0, done, frames=10, НЕТ "WARNING: ... locked frames"

# 4. 20 кадров 10-bit HEVC B=5
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i testdata\bench_src.mkv -e testdata\animejanai_fp16.engine `
  -o "$env:TEMP\smoke10.mkv" --profile quality --frames 20
# ожидаем: yuv420p10le, 20 кадров, CFR, без warning
```

Проверка контейнера:

```powershell
ffprobe -v error -select_streams v:0 -count_frames `
  -show_entries stream=nb_read_frames,avg_frame_rate,pix_fmt -of default=nw=1 out.mkv
```

### 10.3. Покадровое сравнение

Для правок, которые не должны менять пиксели/битстрим (рефакторинг,
оптимизация memset, unlock, batch), сравни `framemd5`:

```powershell
ffmpeg -v error -i old.mkv -map 0:v:0 -f framemd5 old.md5
ffmpeg -v error -i new.mkv -map 0:v:0 -f framemd5 new.md5
Compare-Object (Get-Content old.md5) (Get-Content new.md5)
# ожидаем 0 отличий на той же GPU/драйвере
```

Если правка меняет энкодер/пиксели — используй PSNR/SSIM/MAE и объясни
допуск. Для цвета проверяй saturation/MAE на одном кадре, а не «на глаз».

### 10.4. Пакет

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i testdata\cut.mkv -i testdata\bench_src.mkv `
  -e testdata\animejanai_fp16.engine --out-dir "$env:TEMP\batch" --frames 5
# ожидаем: [batch] итог: ok=2 failed=0, два файла <stem>.mp4
```

### 10.5. Engine из ONNX

Для проверки сборки engine скопируй свою тестовую ONNX во временную папку
(в репозитории моделей нет) и запусти `--onnx`:

```powershell
Copy-Item <путь-к-модели.onnx> "$env:TEMP\model.onnx" -Force
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 `
  -i <путь-к-видео.mkv> --onnx "$env:TEMP\model.onnx" `
  -o "$env:TEMP\onnx.mkv" --frames 1 --rebuild-engine
```

Первый запуск дольше (сборка). Проверь, что появился кэш рядом с ONNX или в
`%LOCALAPPDATA%\NisshokuSR\engines`, и что повторный запуск его подхватывает.
Не используй боевой engine как «тест ONNX»: он уже готовый.

### 10.6. Portable

```powershell
run_build_release.bat
# проверить:
dist\NisshokuSR\nisshoku.exe --help
dist\NisshokuSR\nisshoku.exe --version
# короткий прогон из portable на 5-10 кадрах
```

Проверь наличие builder resource (или ожидаемое поведение с
`NISSHOKU_SKIP_BUILDER=1`), размеры zip/setup, хеши. Не редактируй
`dist/**` вручную.

### 10.7. Ошибки и fault injection

Штатных тестов на ошибки CUDA/NVENC нет. Если трогаешь error path:

* проверь, что ошибка возвращает `false`/`failed`, а не тихий пропуск;
* проверь, что пакетный режим продолжает следующий файл;
* проверь, что временный TS/лог не теряются при ошибке;
* если инъекция невозможна, явно напиши в отчёте, что путь не проверен.

### 10.8. Бенчмарк

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bench.ps1 `
  -In testdata\bench_src.mkv -Engine testdata\animejanai_fp16.engine `
  -Out testdata\bench.h265 -Label bench -ExtraArgs '--profile quality-split'
```

Смотри `summary.txt`, GPU CSV (SM/MEM/ENC/DEC), CPU CSV. `nvidia-smi dmon`:
`enc%` должен грузиться, `copy ~0`. Не сравнивай цифры с другими GPU/версиями
драйвера без оговорки.

---

## 11. Документация: README, CHANGELOG, AGENTS

Это обязательная часть работы, а не «после релиза».

### 11.1. README.md — пользовательское поведение и журнал багов

README — единственный пользовательский источник правды. Он должен содержать:

1. Назначение и архитектуру в одном экране.
2. Требования и точные версии.
3. Сборку.
4. Portable/installer.
5. Движок: engine/ONNX/кэш/rebuild.
6. Запуск: CLI, профили, пакет, коды возврата.
7. Выход: контейнеры, аудио/субтитры, raw, AV1.
8. Цвет.
9. GUI.
10. Модель/бенчмарки.
11. **«Грабли, уже наступили»** — журнал багов.
12. **«Известные ограничения»** — открытые баги/ограничения.
13. Тюнинг/апгрейд, если актуально.

#### Правила журнала багов

* **Каждый** встреченный баг попадает в README, даже если исправлен в той же
  сессии. Не удаляй старые записи; если баг вернулся — добавь новую или
  обнови статус.
* Запись должна быть проверяемой: симптом, причина, исправление/статус,
  как проверить. Без «что-то сломалось».
* Статусы: `Исправлено`, `Открыт`, `Обходной путь`. Для открытого бага
  укажи, что именно не работает и как воспроизвести.
* Если баг внешний (драйвер/TRT/FFmpeg/SDK) — укажи это и версии.
* Формат:

```markdown
N. **Короткое имя.** Статус: исправлено/открыт/обходной путь.
   Симптом: ...
   Причина: ...
   Исправление: ...
   Проверка: ...
```

* Баг, найденный при ревью/агентом, тоже обязателен: README — общая база,
  а не личный блокнот.
* Если баг нельзя воспроизвести без конкретного файла — опиши условия
  (GPU, драйвер, кодек, разрешение, флаг) и не выдумывай путь.

#### Когда обновлять README

* Новый/изменённый флаг, профиль, дефолт, формат выхода.
* Изменение требований/зависимостей/сборки/portable.
* Новый баг, исправление, ограничение, обход.
* Изменение поведения пакета/имён/кодов возврата.
* Изменение цвета/CFR/аудио/субтитров.
* Новые замеры/бенчмарки, если они заявлены как ориентир.

Не пиши в README внутренние детали, которые нужны только агенту: для этого
есть AGENTS.md. README — для пользователя и для баг-лога.

### 11.2. CHANGELOG.md — обязателен

Файл `CHANGELOG.md` в корне. Ведётся на русском, кратко, по бизнес-логике.

Формат:

```markdown
# Changelog

## [0.4.2] — 2026-11-19

### Добавлено

1. ...

### Изменено

1. ...

### Исправлено

1. ...
```

Правила:

* Заголовки — `Добавлено` / `Изменено` / `Исправлено`.
* Пустая строка перед нумерованным списком.
* Кратко: что изменилось для пользователя, без путей к файлам и технических
  деталей.
* Новые версии сверху; до релиза — `## Не выпущено`.
* Обновлять при каждом пользовательском изменении: флаг, поведение, формат,
  исправление, зависимость, сборка.
* Не переписывать историю; старые записи не удалять.
* Если изменение только внутреннее (рефакторинг без поведения) — можно не
  писать, но если оно влияет на стабильность/производительность, лучше указать.

### 11.3. AGENTS.md

Обновляй этот файл, если меняются:

* карта файлов/подсистем;
* команды сборки/запуска/тестов;
* инварианты и контракты;
* известные ловушки и правила сессий;
* процесс релиза/переименования.

Не дублируй сюда полный журнал багов: канонический журнал — README. Здесь
можно держать краткий snapshot и ссылку на README.

### 11.4. Генерируемые документы

* `dist/NisshokuSR/README.md` — копия корневого README на момент сборки.
  Правь корневой README и пересобери release; не редактируй копию.
* `PORTABLE.txt`, `nisshoku.cmd`, `NisshokuSR.iss` — генерируются
  `build_release.py`. Не правь вручную.
* `configs/pipeline.json` — сейчас не читается рантаймом; не считай его
  источником настроек.

### 11.5. `docs/CLI.md`

Обновляй при изменении флагов, дефолтов, профилей, форматов выхода, кодов
возврата или пользовательских примеров. README ссылается на этот файл;
внутренние инварианты и карту файлов сюда не дублируй.

---

## 12. Рабочий процесс агента

### 12.1. Перед правкой

1. Прочитай `AGENTS.md` (этот файл), `README.md` (особенно баги/ограничения),
   `CHANGELOG.md`, `version.txt`, `CMakeLists.txt`.
2. Найди затронутые файлы и существующие решения; не дублируй.
3. Для нетривиальной задачи составь карту: что меняем, где, какие инварианты
   затрагиваем, как проверим.
4. Для багфикса сначала воспроизведи баг или опиши точный сценарий; затем
   правка; затем зелёная проверка.
5. Не угадывай API/флаги/сигнатуры — проверяй по коду, SDK или документации.
6. При изменении CLI прочитай и обнови `docs/CLI.md`.

### 12.2. Во время правки

* Минимальный диф: не рефакторить соседнее, не менять форматирование, не
  трогать линтеры.
* Не оставлять мёртвый код, закомментированные блоки, отладочные принты.
* Не добавлять зависимость без явного согласия.
* Не менять CLI/профили/формат прогресса без обновления README и GUI.
* Не менять цвет/CFR/энкодер «по ощущениям»: только с тестом.
* Не добавлять потоки/контексты/стримы без пересмотра инвариантов.
* Ошибки обрабатывать по существу: `false`/error string, не глотать.
* Ресурсы — RAII; NVDEC unlock ровно один раз; teardown в правильном порядке.

### 12.3. После правки

1. Собери `scripts/build_windows.ps1`.
2. Прогони релевантный smoke/регресс; для изменений пикселей — framemd5/PSNR.
3. Проверь, что нет `WARNING: ... locked frames`, тихих дропов, лишних
   `failed`/`error`.
4. Обнови README (поведение + баг-лог), CHANGELOG, AGENTS при необходимости.
5. В отчёте напиши: что изменено, что проверено, что не проверено и как это
   проверить. Не пиши «работает» без вывода/кода возврата.

### 12.4. Коммиты и пользовательские данные

* Не коммить/пушить без явной просьбы.
* Не удалять `testdata`, `dist`, `setup_output`, `bench_results`, пользовательские
  видео и внешние workflow-папки.
* Для тестов использовать `%TEMP%`/уникальную папку и удалять за собой.
* Не запускать destructive-команды по `H:\` и другим пользовательским дискам.

### 12.5. Если что-то не сходится

* Два неудачных исправления подряд — остановись, пересмотри подход, не чини
  третий раз поверх.
* Если требование неоднозначно (имя, формат, профиль, модель) — спроси.
* Если правка требует изменения инварианта — предложи план и тест, не делай
  молча.

---

## 13. Известные баги и ограничения (snapshot)

Канонический список — README, разделы «Грабли» и «Известные ограничения».
Здесь краткий снимок; при изменении статуса обнови оба места.

### 13.1. Открытые

| Проблема | Суть | Где |
|---|---|---|
| `--frames` + Opus | Аудио обрезается неточно (~4.478 с вместо ~2.5 с на тесте) | README, «Грабли» |
| Full-range/BT.601 | Не поддерживаются, вход считается limited BT.709 | README, «Цвет» |
| AV1 в MKV/MP4 | Промежуточный TS не несёт AV1; только raw | README, «Выход» |
| Raw при B>0 | Теряется порядок PTS; raw только B=0 | README, «Выход» |
| Engine на другой GPU | Engine привязан к GPU/TRT; нужен `--onnx` | README, «Движок» |
| Portable без builder | Не умеет собирать engine из ONNX | README, «Portable» |
| GUI в portable | Python/Tkinter не входит | README, «GUI» |
| 3-тредовый конвейер | Не реализован; синк на кадр | README, «Что внутри» |
| Decode-пакеты | Битые пакеты пропускаются, есть счётчик | README, «Пакет» |
| `configs/pipeline.json` | Не читается рантаймом | AGENTS, раздел 5 |
| `--live` | Зарезервирован | README, help |

### 13.2. Исправленные (не повторять)

| Проблема | Причина | Проверка |
|---|---|---|
| FP32 I/O engine | Конвейер ждёт FP16, чтение за буфером | engine загружается, нет illegal access |
| OOB-чтение TRT | Нужен slack 8 МиБ + зануление хвоста | compute-sanitizer, стабильный выход |
| Отдельные CUDA-контексты | Ломают `cuvidMapVideoFrame`/TRT | один primary-контекст |
| AVCC без BSF | NVDEC ждёт Annex-B, 0 кадров | BSF в Demuxer |
| P010 LSB/MSB | 10 бит в старших битах | цвет не темнеет в 64 раза |
| Locked NVDEC frames | Не было unlock на успехе | 100 кадров без warning |
| Тихий дроп NVENC | `return true` на ошибках | failed вместо дырки |
| Полный memset I/O | Лишние ~59 МиБ/кадр | framemd5 совпал |

---

## 14. Troubleshooting

| Симптом | Вероятная причина | Действие |
|---|---|---|
| `0xc0000135` при запуске | Нет DLL в `PATH` (TRT/FFmpeg/CUDA) | Запускай через `scripts/run.ps1`; TRT DLL в `lib`, не `bin` |
| `NV_ENC_ERR_MAP_FAILED`, `cuvidMap INVALID_*` | Не тот CUDA-контекст, FP32 I/O, OOB | Проверь один primary-контекст, FP16 I/O, slack |
| 0 кадров из MP4/MKV | Нет BSF AVCC->Annex-B | Проверь `Demuxer` BSF для H.264/HEVC |
| Цвет уехал | Full/limited, BT.601, P010 MSB, chroma | Проверь pre/post, VUI, P016/MSB |
| `inf`/`nan`/чёрный выход | Мусор в slack, FP32 I/O, битый engine | Занули slack, проверь IO, пересобери engine |
| `WARNING: N locked frames` | Нет unlock на успехе/ошибке | Проверь путь `processFrame`, RAII |
| VFR в MP4 | Округлённый fps/таймскейл | Точный rational, `video_track_timescale` |
| Аудио разъехалось | `start_time`/`itsoffset`/PTS-дырка | Проверь `start_us`, `setts`, нет дропов |
| Engine не грузится на другой GPU | Привязка к GPU/TRT | `--onnx`, пересборка |
| ONNX с кириллицей не парсится | Парсер ждёт ACP-narrow | Латинский путь или переименуй папку |
| Нет `nvinfer_builder_resource_10.dll` | Portable без builder | Добавь DLL или используй готовый engine |
| `ffmpeg.exe не найден` | Нет рядом с exe/`FFMPEG_ROOT` | Положи `ffmpeg.exe` рядом или задай env |
| H.264 10-bit | H.264 не поддерживает 10-bit | Используй HEVC |
| Split + H.264 | Не поддерживается | HEVC/AV1 |
| `--frames` + Opus | Известный баг обрезки | Не использовать для аудио; см. README |
| `num_encoders` стал 1 | B/lookahead или split | Это ожидаемое правило |
| `--rebuild-engine` без ONNX | Нет источника сборки | Добавь `--onnx` |
| `configs/pipeline.json` не влияет | Не читается рантаймом | Меняй CLI/профиль, не JSON |

---

## 15. Чек-лист переименования проекта

Если имя меняется снова, пройди **все** пункты; иначе останутся старые пути,
AppId, кэш и внешние скрипты.

1. **CMake/бинарь**
   * `project(...)` в `CMakeLists.txt`;
   * имя target/exe;
   * `NISSHOKU_VERSION` и `--version`;
   * `build/` output.
2. **Скрипты**
   * `scripts/build_windows.ps1` (пути к exe);
   * `scripts/run.ps1` (имя exe);
   * `scripts/bench.ps1` (имя exe, regex `done, frames=`);
   * `run_gui.bat`, `tools/gui.py` (пути к build/dist).
3. **Release**
   * `build_release.py`: `PORTABLE_NAME`, имена zip/setup, README, `.cmd`;
   * `version.txt`;
   * генерируемый `.iss`: `AppId`, `AppName`, `DefaultDirName`,
     `OutputBaseFilename`, ярлыки, uninstall-имя.
4. **Кэш/данные**
   * `%LOCALAPPDATA%\NisshokuSR\engines` — решить: мигрировать или оставить
     старый путь;
   * env `NISSHOKU_*` — сохранить совместимость или задокументировать breaking
     change;
   * `NISSHOKU_THIRD_PARTY`, `NISSHOKU_SKIP_BUILDER`, `NISSHOKU_KEEP_TS`.
5. **GUI/внешние интеграции**
   * заголовок окна, дефолтные пути;
   * внешние workflow/скрипты, которые вызывают exe по имени;
   * ярлыки, scheduled tasks, bat-файлы пользователя.
6. **Документы**
   * README, CHANGELOG, AGENTS, PORTABLE.txt (генератор);
   * примеры команд и пути.
7. **Проверка**
   * поиск старого имени по всему репозиторию и скриптам;
   * сборка, smoke, пакет, portable, installer;
   * проверить, что старый AppId не ломает обновление (решить осознанно).
8. **Совместимость**
   * если пользователи уже используют CLI/engine-кэш — оставить alias/wrapper
     или миграцию; иначе это breaking change, отразить в CHANGELOG.

---

## 16. Definition of Done

Правка считается готовой, только если:

- [ ] Проект собирается `scripts/build_windows.ps1` без новых ошибок.
- [ ] Релевантный smoke/регресс пройден; для пикселей — framemd5/PSNR;
      для CFR — ffprobe; для пакета — `ok=N failed=0`.
- [ ] Нет новых `WARNING: ... locked frames`, тихих дропов, NaN/illegal access.
- [ ] Инварианты из раздела 2 не нарушены.
- [ ] README обновлён: поведение + баг-лог/ограничения.
- [ ] CHANGELOG обновлён.
- [ ] AGENTS.md обновлён, если изменилась карта/команды/инварианты.
- [ ] `version.txt` поднят, если это релиз.
- [ ] Генерируемые артефакты не правились вручную; release пересобран, если
      обещан portable/installer.
- [ ] В отчёте есть: что изменено, что проверено, что не проверено и как
      проверить.

---

## Приложение A. Текущее состояние (снимок)

* Версия в `version.txt` — 0.4.1; `dist/` и installer могут быть старее
  исходников после локальных правок. Всегда проверяй дату/версию артефакта.
* Бинарь `build/nisshoku.exe` — dev-сборка; portable — только после
  `run_build_release.bat`.
* `testdata/` и готовые engine в исходный репозиторий не входят: это локальные
  файлы, исключённые через `.gitignore`. Для прогонов нужен свой engine
  (или `--onnx`).
* GUI — dev-only; в portable Python нет.
* Основной рабочий профиль для качества — `quality`/`quality-split`; быстрый —
  `fast`/`balanced`.

Этот снимок устаревает; если он расходится с README/version.txt — правы они.
