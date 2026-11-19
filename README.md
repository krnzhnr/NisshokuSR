# NisshokuSR — NVDEC -> TRT-апскейл -> NVENC, только GPU, под Windows

Цель: кадр рождается на GPU и умирает на GPU. Ноль `DtoH/HtoD` в steady-state.

> Для разработчиков и агентов: `AGENTS.md` — карта проекта, инварианты и
> процессы; `CHANGELOG.md` — история изменений; `docs/CLI.md` — полный
> CLI-справочник. Баги и ограничения ведутся ниже, в разделах «Грабли» и
> «Известные ограничения».
>
> В репозитории нет тестовых видео, моделей и готовых engine: папка
> `testdata/` исключена через `.gitignore`. Для локальных прогонов положи
> свои файлы в `testdata/`; в коммит они не попадут.

## Что внутри

```
IO (libavformat + h264_mp4toannexb BSF) -> NVDEC (SDK NvDecoder, CUDA)
 -> pre-kernel NV12/P016->RGB planar FP16 (CUDA, fused)
 -> TensorRT enqueueV3 (FP16 static engine, тот же поток)
 -> post-kernel RGB->NV12 или P010 (CUDA, BT.709)
 -> NVENC (SDK NvEncoderCuda, P7 HQ, Main10, chunked dual/split) -> mux/сеть
```

Один управляющий тред: decode -> compute -> encode -> mux. Трёхтредовый
разнос не реализован. Синхронизация — через один CUDA-стрим, без лишних
`cudaDeviceSynchronize` в цикле.

## Требования

* Windows 11 x64, драйвер свежий, HAGS ON, Game Bar OFF.
* CUDA Toolkit 12.8/12.9 (12.x, НЕ 13.x — под него собран TRT 10.8),
  TensorRT 10.8 (ветка cuda-12.x, zip), Video Codec SDK 13.1+, FFmpeg shared (BtbN lgpl-shared).
* MSVC v143 (VS2022): nvcc 12.8 падает на компиляторе VS18/2026 (`cudafe` крэшится,
  `-allow-unsupported-compiler` не спасает). Доставь workload v143 через VS Installer.
* Env: `NVCODEC_SDK_DIR`, `TRT_ROOT`, `FFMPEG_ROOT` (user-level `setx`, затем НОВЫЙ терминал).
* Бинарь собирается под sm_75/86/89/120 — RTX 20/30/40/50. TensorRT engine
  по-прежнему привязан к конкретной GPU: готовый подхватывается, иначе
  собирается из ONNX при запуске.

## Сборка

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Ninja + portable cmake/ninja из `third_party` (VS-генератор не работает:
у CUDA 12.8 нет `.props` для VS18). Бинарь: `build/nisshoku.exe`.

## Portable и installer

Один батник — сборка, portable-папка, ZIP и Inno Setup installer:

```powershell
run_build_release.bat
```

Он вызывает `build_release.py` (Python-промежуток, как в k-tools), версия берётся
из `version.txt`. Результат:

```text
dist\NisshokuSR\                            — автономная portable-папка
dist\NisshokuSR_v<version>_portable.zip     — ZIP
setup_output\NisshokuSR_v<version>_Setup.exe — installer
```

В portable лежат exe, `cudart64_12.dll`, `nvinfer_10.dll` (+ plugin/dispatch),
`nvonnxparser_10.dll` и `nvinfer_builder_resource_10.dll` (сборка engine из ONNX),
FFmpeg DLL + `ffmpeg.exe` (финальная упаковка MKV/MP4), VC++ runtime,
`README.md` и `docs\CLI.md`.
Готовый `engines\animejanai_fp16.engine` копируется только если он есть локально
в `testdata`; в исходном репозитории моделей и engine нет — используйте `--onnx`
или положите свой engine. CUDA Toolkit, TensorRT SDK и Video Codec SDK
на машине пользователя не нужны — только драйвер NVIDIA. exe работает на
RTX 20/30/40/50 (sm_75/86/89/120); любой готовый engine привязан к GPU/TRT.

`nvinfer_builder_resource_10.dll` весит около 2 ГБ и нужен только для сборки
engine из ONNX. Если он не нужен, собери portable с `NISSHOKU_SKIP_BUILDER=1`:
тогда работают только готовые `.engine`. Из-за FFmpeg и builder-ресурса
папка/zip/installer заметно тяжелее.

Если Inno Setup не найден, portable и ZIP всё равно собираются, а installer
пропускается. Путь к `third_party` можно переопределить переменной
`NISSHOKU_THIRD_PARTY`.

### Что делает установщик

* Ставит приложение per-user, без прав администратора; по умолчанию — в
  `%LOCALAPPDATA%\Programs\NisshokuSR`.
* Копирует exe, CUDA runtime, TensorRT DLL (включая builder resource), FFmpeg
  DLL + `ffmpeg.exe`, VC++ runtime и `nisshoku.cmd`.
* Модель и готовый engine **не входят**: добавьте свой engine или ONNX.
  С builder resource сборка engine из ONNX работает после установки.
* Создаёт ярлык «NisshokuSR CLI»: открывает командную строку в папке установки
  и печатает help. `PATH` не меняется, поэтому из чужого терминала `nisshoku`
  не появится — запускайте через ярлык, `nisshoku.cmd` или полный путь.
* GUI в установщик не входит (это dev-инструмент).
* Установщик и exe не подписаны: SmartScreen может показать предупреждение.

## Движок апскейла

CLI принимает либо готовый `.engine` (`-e/--engine`), либо ONNX (`--onnx`).
Если передан только ONNX, порядок такой:

1. рядом с ONNX ищется `<stem>.engine` (готовый движок);
2. затем кэш `<stem>_<W>x<H>_<gpu>_fp16_trt<версия>.engine` рядом с ONNX;
3. если ничего не подошло — движок собирается из ONNX под текущую GPU и
   разрешение (может занять от десятков секунд до нескольких минут) и
   сохраняется в кэш. Если папка с ONNX недоступна для записи, кэш уходит в
   `%LOCALAPPDATA%\NisshokuSR\engines`.
4. `--rebuild-engine` игнорирует готовый движок и кэш (требует `--onnx`).
   В пакете пересборка делается один раз на пару «ONNX + разрешение + GPU».

Кэш проверяет путь/размер/mtime ONNX, GPU, версию TensorRT и разрешение входа.
Движок, собранный под другое разрешение или GPU, не подхватывается: CLI
пересоберёт его из ONNX. Для сборки нужны `nvonnxparser_10.dll` и
`nvinfer_builder_resource_10.dll`; при запуске готового engine они не нужны
(кроме onnxparser, который импортирует сам exe).

Ручная сборка (для отладки):

1. `$env:PYTHONUTF8="1"; python tools/export_onnx.py --arch bilinear --dtype fp16 --out model.onnx --h 1080 --w 1920 --scale 2`
   (`--arch tiny` без `--ckpt` = случайные веса = шум на выходе, только для проверки графа;
   PYTHONUTF8 обязателен — логгер torch падает на cp1251-консоли).
2. `tools/build_engine.bat` (`trtexec --fp16 --stronglyTyped --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw`).
   **Формат IO строго FP16 с обеих сторон**: по дефолту trtexec оставляет FP32-IO,
   и тогда инференс читает в 2 раза больше памяти, чем выделено, — тихая порча
   всего конвейера ( illegal access → NV_ENC_ERR_MAP_FAILED, cuvidMap INVALID_* ).
   Загрузчик (`TrtUpscaler::init`) сверяет тип биндингов. При сборке из ONNX
   FP32-вход/выход принудительно переводится на FP16; если это невозможно,
   сборка завершится ошибкой с подсказкой.
3. ONNX может быть статическим (вход ровно под разрешение видео) или
   динамическим. Для динамического входа при сборке фиксируется профиль
   `[1,3,H,W]`; для другого разрешения будет отдельный кэш.
4. Для скорости бери легкие сети: ESPCN-S / FSRCNN-S / anime-6B. Полный RealESRGAN_x4plus в 60fps не влезет.
5. Боевые веса (Real-ESRGAN и т.п.): `export_onnx.py --arch tiny --ckpt model.pth`
   (граф тот же, веса твои), дальше по шагу 2.

## Запуск

CLI удобнее запускать через `scripts/run.ps1` — он сам кладет в PATH `TRT lib+bin`, `FFmpeg bin`,
`CUDA bin` (без них exe падает с `0xc0000135`, причем TRT-DLL лежат в `lib`, а не в `bin`).
Для тестов есть простое окно: `run_gui.bat` (см. ниже).

```powershell
# Справка/профили/сухой прогон
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --help
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --list-profiles
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 --dry-run --profile quality-split

# quality-профиль: Main10, B=5, ref=8, LA=16, CQP18, одна сессия
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.mkv --profile quality
# То же + split encoding на два NVENC (быстрее)
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.mkv --profile quality-split
# Модель в ONNX: готовый engine подхватится, иначе соберётся под видео/GPU
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv --onnx model.onnx -o out.mp4 --profile quality-split
# Пакет: папка целиком -> <папка>\upscaled\<stem>.mp4
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i "D:\серии" --onnx model.onnx --profile quality-split
# Пакет: несколько файлов в свою папку, MKV
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i a.mkv -i b.mp4 --out-dir "D:\out" --out-ext mkv --onnx model.onnx --profile quality-split
# Быстрый P1/B=0/dual 8-bit
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.h265 --profile fast
# 8-бит поверх quality-профиля
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.mkv --profile quality --bit-depth 8
# Диагностика без compute
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o gray.h265 --profile fast --gray --frames 60
# Упаковка в MKV: видео + аудио + субтитры из исходника
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.mkv --profile quality-split
# То же в MP4 (субтитры ASS/PGS в MP4 не копируются)
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -i in.mkv -e model.engine -o out.mp4 --profile quality-split
```

CLI: `--help`, `--version`, `--list-profiles`, `--dry-run`. Профили
`fast|balanced|quality|quality-split`, явные флаги перекрывают профиль
независимо от порядка. Коды возврата: `0` успех,
`1` ошибка выполнения (хотя бы один файл пакета), `2` неверные аргументы.

Полный справочник по флагам, форматам, пакетному режиму и ограничениям —
[`docs/CLI.md`](docs/CLI.md).

Входы: `-i/--in` можно повторять, позиционные пути тоже считаются входами.
Путь-папка разворачивается в видеофайлы верхнего уровня (`.mkv .mp4 .m4v .mov
.ts .m2ts .webm .avi .wmv .mpg .mpeg .flv`) и сортируется по имени. Пакет
обрабатывается последовательно; ошибка одного файла не останавливает остальные,
итог печатается как `[batch] итог: ok=N failed=M`. Ошибка NVENC на кадре не
превращается в тихий пропуск: файл помечается failed (дырка по PTS хуже явной
ошибки), пакет продолжает со следующего файла. Битые пакеты декодера
по-прежнему пропускаются, но в конце печатается `[dec] ошибок декодирования: N`.

Выход в пакетном режиме: `<папка входа>\upscaled\<stem>.<ext>`, по умолчанию
`.mp4`, имя повторяет исходник. Папку можно задать через `--out-dir`, формат —
через `--out-ext mp4|mkv|h264|h265|av1|raw`. Если выход совпал с входом или два
входа дают одно имя, CLI останавливается с ошибкой до запуска. Для одного входа
по-прежнему можно указать файл через `-o`.

`--profile quality` — перенос настроек NVEncC из SVFI: `-u quality` (P7/HQ), `--cqp 18`,
`--gop-len 250`, `-b 5`, `--ref 8`, `--lookahead 16`,
`--bref-mode middle`, `--tier main`, **HEVC Main10 / P010**. B-кадры и lookahead
держат кадры внутри сессии, поэтому chunked dual-NVENC в этом режиме
автоматически отключается (1 сессия); для `B=0` двойной NVENC остается.
`--bit-depth 8` переключает выход обратно на NV12 8-bit.

`--split-enc forced_2` — split encoding NVENC: один кадр делится между двумя
движками в одной сессии. На 5070 Ti с `--profile quality` на чистом CPU это даёт
**67.9 fps против 45.6 без split** при том же битрейте (+49%). H.264 split не
поддерживает.
В SVFI тот же режим включается через
`--force-args "render_customized=--split-enc||forced_2"`, но там упор в
VSPipe/SR (enc 12–15%), поэтому выигрыш почти нулевой.

Выход определяется расширением:

* `.mkv` — видео + аудио + субтитры и метаданные из исходника (копирование);
* `.mp4` — видео + аудио; субтитры ASS/PGS в MP4 не копируются;
* `.h265/.h264/.av1` — сырой поток без звука; AV1 в MKV/MP4 пока не
  поддерживается (промежуточный MPEG-TS его не несёт). Raw годится только
  для профилей без B-кадров (`fast`, `balanced`) — у `quality*` B=5, и raw-демуксер
  теряет порядок PTS.

Пайплайн пишет временный MPEG-TS (libavformat), затем переупаковывает его
вместе с исходником через встроенный `ffmpeg.exe`: аудио и субтитры копируются
без перекодирования, ручной hvcC не нужен. Частота кадров берётся из исходника
точным rational (например, `24000/1001`), а для MP4 задаётся
`video_track_timescale`, поэтому выход остаётся CFR, а не VFR. Для B-кадров MP4
получает edit list, который убирает задержку переупорядочивания; raw-демуксер
FFmpeg 8.1 такие PTS не восстанавливает, поэтому raw для B>0 по-прежнему не
годится. Отключение дорожек: `--audio none`, `--subs none`. Для разбора remux
переменная `NISSHOKU_KEEP_TS=1` оставляет временный `<out>.tmp.ts` рядом с
результатом. `--frames N` дополнительно обрезает аудио/субтитры по последнему
кадру (для Opus обрезка сейчас неточная — известный баг, см. «Грабли»); без
лимита они копируются целиком, включая хвост исходника.

## Цвет

Вход и выход — **BT.709 limited (tv)**, внутри пайплайна — **full RGB**.
NVENC VUI пишет BT.709/tv. Pre/post ядра конвертируют limited↔full по BT.709
(не BT.601), chroma при выходе усредняется 2×2. Источники предполагаются
limited BT.709; full-range и BT.601 пока не поддерживаются.

## GUI для тестов

```powershell
run_gui.bat
```

или `python tools\gui.py`. Окно запускает `build\nisshoku.exe` (если его нет —
`dist\NisshokuSR\nisshoku.exe`), показывает вывод дочернего процесса и даёт
менять контейнер, профиль и параметры NVENC. По умолчанию подставлен
`quality-split` (HEVC 10-bit, B=5, split forced_2).

GUI поддерживает пакетный режим: список входных файлов и папок, отдельное поле
ONNX-модели, необязательный engine, папка выхода и контейнер (по умолчанию MP4).
Выходные имена строятся как `<stem>.<ext>`. Для MP4 субтитры отключаются
автоматически. GUI — dev-инструмент, в portable Python не входит.

CLI каждые 2 секунды печатает
`[progress] frames=… total=… pct=… fps=… speed=… elapsed=… eta=…`; в пакетном
режиме добавляются `file=N files=M`. GUI парсит эти строки и показывает полосу,
fps, ETA и номер файла.

## AnimeJaNai (реальная SR-модель, проверено)

Модель: `AnimeJaNai_V5.1_Compact` — нативно FP16, динамические H/W.
Сборка движка (рецепт как у SVFI, но со static shape под 1080p):

```powershell
trtexec --onnx=AnimeJaNai_V5.1_Compact_225000.onnx --saveEngine=animejanai_fp16.engine \
  --shapes=input:1x3x1080x1920 --fp16 --tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN \
  --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --builderOptimizationLevel=3 \
  --memPoolSize=workspace:128
```

Код сам добивает `setInputShape` для динамических дименшенов (`TrtUpscaler`).
Важно: IO строго FP16 с обеих сторон — иначе см. «Грабли».

Замер на том же файле (1080p23.976, 34046 кадров, 5070 Ti, dual NVENC):
наш конвейер против SVFI (VapourSynth + y4m + NVEncC, CQP 18, quality, 10 бит):

| | SVFI | NisshokuSR |
|---|---|---|
| Апскейл | TRT AnimeJaNai | TRT AnimeJaNai (тот же .onnx) |
| Энкодер | NVEncC H.265 Main10, quality, CQP 18, B=5, lookahead 16 | встроенный NVENC H.265: `--profile quality` = P7/HQ, CQP 18, B=5, ref=8, lookahead 16, GOP 250; без B — P1/HQ, dual |
| Скорость, end-to-end | ~42 fps (801 c на эпизод) | ~58 fps (B=0 dual, старый замер); `--profile quality` — ~44.7 fps e2e (45.6 pipeline), `--profile quality-split` — **~64.7 fps e2e** (67.9 pipeline) на 815-кадровом фрагменте |
| Выход | ~3.7 ГБ (~21 Мбит/с), 10 бит | B=0: ~22 Мбит/с 8-bit; `--profile quality`: **~14.6 Мбит/с, HEVC Main10**, split — столько же |

Наш выигрыш — отсутствие VapourSynth/y4m-прослойки и DtoH-копий: все живет на GPU
от NVDEC до NVENC. 10-битный тракт теперь свой: P010 + HEVC Main10 + BT.709;
на тестовом фрагменте SSIM против SVFI — 0.995 по Y, битрейт 14.60 против
14.67 Мбит/с.

Профиль `--profile quality` повторяет NVEncC один-в-один, включая Main10/P010.
B-кадры/lookahead держат кадры внутри сессии, поэтому chunked dual-NVENC в этом
режиме отключается. На чистом CPU без split одна сессия даёт 45.6 fps, с
`--split-enc forced_2` — 67.9 fps (+49%) при том же битрейте. B=0 dual остаётся
запасным вариантом. `--bit-depth 8` — если нужен старый NV12 8-bit.

## Двойной NVENC на 5070 Ti (2x Gen9, 1x декодер)

* Карта: [2 энкодера Gen9 + 1 декодер Gen6](https://www.pugetsystems.com/blog/2025/01/21/nvidia-geforce-rtx-50-series-features-for-content-creators) — декодер один, его не распараллелить, а энкодеры да. Второй источник: [5070 Ti — 2 энкодера, 5070 — 1](https://boxx.com/blog/nvidia-rtx-50-gpus-media-workflows).
* Режим — `chunked GOP` (`NvEncWrapper.cpp`): четные GOP на enc0, нечетные на enc1,
  чанк = GOP, первый кадр чанка — принудительный IDR. Референсы не рвутся,
  в отличие от even/odd-кадров по двум энкодерам.
* С B-кадрами/lookahead chunked-режим несовместим: хвост чанка живет в очереди
  сессии до следующего кадра или EOS. Код принудительно ставит 1 сессию
  (`--profile quality`); для B=0 dual остается.
* `--split-enc forced_2` — другой способ занять оба движка: один кадр делится
  между ними внутри одной сессии. Работает с B-кадрами и Main10; H.264 не
  поддерживает. На 5070 Ti это быстрее chunked dual в режиме quality.
* Порядок при асинхронном выводе держит reorder-буфер по глобальному PTS
  (окно = max задержки NVENC), плюс `nExtraOutputDelay` по умолчанию.
* Проверка: `nvidia-smi dmon` — `enc%` должен грузиться, `copy ~0`. Для B=0
  порядок в файле — по PTS; для B>0 — decode-order (POC), переставлять нельзя.

## Грабли, уже наступили (не повторять)

1. **TRT IO dtype.** Движок без явных `--inputIOFormats/--outputIOFormats` оставляет FP32-IO.
   Конвейер подает/ждет FP16 → чтение/запись мимо буферов → illegal access,
   который травит ВЕСЬ контекст (падает и NVENC, и NVDEC, хотя они ни при чем).
2. **TRT читает за концом входа** (~4.2 МБ окна CUTENSOR-permute на Blackwell 12.x).
   I/O-буферы с запасом 8 МБ, на кадр зануляется только slack-хвост
   (`TrtUpscaler::inDataBytes/outDataBytes`): сами тензоры пишут pre-kernel и engine.
   Без запаса — те же симптомы, что в п.1. Проверено compute-sanitizer.
3. **Один CUDA-контекст на всех** (driver-созданный primary). Отдельные контексты
   на декодер/энкодер + `cuCtxPopCurrent/SetCurrent` вокруг ломают
   `cuvidMapVideoFrame` (INVALID_HANDLE/INVALID_VALUE) — инструментально доказано
   эталонным AppDec (у него тоже один контекст).
4. **MP4/MKV отдает AVCC-пакеты, NVDEC ждет Annex-B.** Обязателен BSF
   `h264_mp4toannexb` / `hevc_mp4toannexb` в демuxer'е, иначе 0 кадров без ошибок.
5. **WDDM + `cudaMemcpy DtoH` в hot-loop лучше не трогать** — отладочные чтения
   на хост могут врать/сыпать; весь мониторинг — через device-side счетчики.
6. **NVENC P1/ULL + CBR работает как часы**, но только если конфиг 1-в-1 как у семплов
   (см. `NvEncWrapper::init`); `vbInitial*`/`maxBitrate` не трогать (оставить 0).
   CQP/P7 + B/lookahead тоже проверен, но требует `num_encoders=1` (см. выше).
7. **P010 в NVENC — 10 бит в СТАРШИХ битах 16-битного слова.** Если записать
   10-битное значение как LSB (как в yuv420p10le), картинка станет темнее в 64 раза.
   NVDEC P016 читается так же (MSB), см. `kP016ToRgbPlanar`/`kRgbPlanarToP010`.
8. **NVDEC-кадр оставался locked после успешной обработки.** Симптом:
   `WARNING: N locked frames were not unlocked` при shutdown, где N равно числу
   кадров. Причина: `unlock.release()` разоружал RAII, но явного `dec.unlock()`
   на успешном пути не было. Исправлено: кадр разблокируется сразу после
   DtoD-копии в буфер энкодера; RAII остаётся только на ошибки/исключения.
   Проверка: 100 кадров — warning отсутствует.
9. **Тихий дроп кадра NVENC.** Симптом: `skip frame`, файл завершается успешно,
   но остаётся дырка по PTS; при B=0 reorder-буфер мог держать хвост до 64
   кадров. Аудио не «уезжает» (оно ремуксится со своими PTS), но кадр
   залипает/пропадает. Причина: `return true` на ошибках копии, `EncodeFrame`
   и `flush`, а также пропуск дырки в `drainContiguous`. Исправлено: копия
   повторяется один раз; ошибка кодирования/флаша/дырка — fatal для файла,
   пакет продолжает со следующего. Проверка: обычные прогоны совпадают
   покадрово; инъекция ошибки NVENC автоматически не тестировалась.
10. **Полный memset I/O на каждом кадре.** Симптом: для 1080p→4K на кадр
    приходилось ~75 МиБ записи, из них ~59 МиБ лишних. Причина: занулялись
    целиком вход и выход, хотя pre-kernel и engine пишут сами тензоры.
    Исправлено: зануляется только slack-хвост 8 МиБ на буфер через
    `cudaMemsetAsync` в `comp_stream`. Проверка: `framemd5` 8-bit и 10-bit
    совпал с версией до оптимизации.
11. **`--frames` + Opus: неточная обрезка аудио. Открыт.** На тестовом файле
    с Opus `--frames 100` даёт ~4.478 с аудио вместо ~2.5 с. Причина не
    локализована: `-t` + `-copyts` и Opus pre-skip/edit list. `--frames` —
    диагностический флаг; для проверки аудио-обрезки его использовать нельзя.
12. **`-o` с расширением, не совпадающим с кодеком.** `-o out.h265` без
    `--codec-out hevc` писал H.264 в файл `.h265`, `-o out.av1` без AV1 —
    H.264 в `.av1`. Исправлено: `ParseCli` проверяет расширение `-o` и
    возвращает код 2 до запуска. `.raw`/без расширения — сырой поток, кодек
    задаёт `--codec-out`.
13. **Raw-выход при B-кадрах.** `--profile quality -o out.h265` раньше запускался,
    хотя raw-демуксер теряет порядок PTS при B>0. Исправлено: CLI возвращает
    код 2, если выбран raw-выход и `bframes > 0`; для `quality*` используйте
    `.mkv`/`.mp4` или явно `--bframes 0`.

## Известные ограничения

* Full-range и BT.601 источники не поддерживаются: вход/выход считаются
  limited BT.709.
* AV1 в MKV/MP4 не поддерживается (промежуточный MPEG-TS его не несёт);
  только raw `.av1`.
* Raw-выход годится только для профилей без B-кадров: у `quality*` B=5, raw
  теряет порядок PTS. CLI теперь запрещает raw при `bframes > 0`.
* TensorRT engine привязан к GPU и версии TensorRT; готовый engine из portable
  собран под RTX 5070 Ti. На другой карте нужен `--onnx` или свой engine.
* Portable без `nvinfer_builder_resource_10.dll` умеет только готовые engine.
* GUI (`tools/gui.py`) — dev-инструмент, в portable Python не входит.
* Трёхтредовый конвейер не реализован: один управляющий тред и синхронизация
  на кадр.
* Битые пакеты декодера пропускаются (это политика для повреждённого входа),
  в конце печатается счётчик `[dec] ошибок декодирования`.
* `configs/pipeline.json` сейчас только упаковывается в portable и не читается
  рантаймом; менять его для изменения поведения бесполезно.
* `--live` зарезервирован и не реализует отдельный live-режим.

## Тюнинг под максимум

* NVENC: `P1, ULL, B=0` live; файл — `--profile quality` (P7/HQ, CQP18, B=5, ref=8,
  lookahead 16, GOP 250, bref-middle) как у SVFI, но одной сессией; либо
  P1/HQ, CQP18, B=0, dual — быстрее на ~20%, битрейт выше.
  На 5070 Ti профиль quality лучше всего идёт с `--split-enc forced_2`: оба
  NVENC на один поток, 67.9 fps и низкий битрейт (на чистом CPU).
* Профиль: Nsight Systems (gaps между кернелами = CPU sync), `nvidia-smi dmon`,
  `scripts/bench.ps1` (per-core CPU + SM/MEM/ENC/DEC/JPG/OFA).
* Налог WDDM на GeForce ~5-15% против Linux, лечится меньшим числом sync.

## Лицензия

Код проекта — MIT, см. `LICENSE`. Сторонние компоненты не покрываются MIT:
FFmpeg (LGPL v3), NVIDIA CUDA / TensorRT / Video Codec SDK идут по своим
лицензиям. При публикации portable/installer добавь их тексты и уведомления.
Модели, веса и готовые engine в репозиторий не входят.

## Апгрейд до честного zero-copy (как AppTransZeroCopy)

Сейчас между decode и encode один `DtoD`-копипаст во входной буфер энкодера (быстро, ~0.2мс).
Следующий шаг — общий `CUarray` с `CUDA_ARRAY3D_VIDEO_ENCODE_DECODE` как output декодера и input энкодера.
