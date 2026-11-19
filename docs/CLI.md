# NisshokuSR CLI — справочник

> Это пользовательская документация по CLI. Установка/portable/движок —
> в [README.md](../README.md). Внутренняя карта проекта и инварианты —
> `AGENTS.md` в исходном репозитории. Если поведение кода и этого файла
> расходятся, прав код; обновите документ.

## Содержание

1. [Что это](#что-это)
2. [Быстрый старт](#быстрый-старт)
3. [Синопсис](#синопсис)
4. [Входы и выходы](#входы-и-выходы)
5. [Флаги](#флаги)
6. [Профили](#профили)
7. [Пакетный режим](#пакетный-режим)
8. [Форматы и матрица выходов](#форматы-и-матрица-выходов)
9. [Движок: engine, ONNX и кэш](#движок-engine-onnx-и-кэш)
10. [Аудио и субтитры](#аудио-и-субтитры)
11. [Диагностика и ограничения](#диагностика-и-ограничения)
12. [Прогресс и коды возврата](#прогресс-и-коды-возврата)
13. [Переменные окружения](#переменные-окружения)
14. [Примеры](#примеры)
15. [Известные ограничения](#известные-ограничения)

---

## Что это

`nisshoku` — Windows-CLI для офлайн-апскейла видео на NVIDIA GPU:

```text
NVDEC -> CUDA pre-kernel -> TensorRT -> CUDA post-kernel -> NVENC -> FFmpeg mux
```

CLI не содержит моделей и готовых engine. Для работы нужен один из вариантов:

* готовый TensorRT engine: `-e/--engine`;
* ONNX-модель: `--onnx`, тогда engine подхватится из кэша или соберётся
  под текущую GPU/разрешение (нужен `nvinfer_builder_resource_10.dll`,
  в portable/installer он есть по умолчанию).

Нужен только драйвер NVIDIA. CUDA Toolkit, TensorRT SDK, Video Codec SDK,
FFmpeg SDK и Visual Studio для запуска не требуются — runtime-DLL уже лежат
рядом с exe.

GUI (`tools/gui.py`) — отдельный dev-инструмент, в installer/portable не входит.

---

## Быстрый старт

```powershell
nisshoku --help
nisshoku --version
nisshoku --list-profiles
nisshoku --dry-run --profile quality-split
```

Один файл с готовым engine:

```powershell
nisshoku -i in.mkv -e model.engine -o out.mkv --profile quality-split
```

Один файл из ONNX (engine соберётся/подхватится автоматически):

```powershell
nisshoku -i in.mkv --onnx model.onnx -o out.mp4 --profile quality
```

Пакет: папка целиком в `out`:

```powershell
nisshoku -i "D:\серии" --onnx model.onnx --out-dir "D:\out" --out-ext mkv
```

После установки ярлык «NisshokuSR CLI» открывает командную строку в папке
установки и печатает help. `PATH` установщик не меняет; `nisshoku` доступен
из этого окна, через `nisshoku.cmd` или по полному пути.

---

## Синопсис

```text
nisshoku -i <вход> -e <engine> -o <выход> [опции]
nisshoku -i <вход> --onnx <model.onnx> -o <выход> [опции]
nisshoku --profile <name> -i <файл|папка> ... --out-dir <dir> [опции]
nisshoku --dry-run [опции]
nisshoku --help | --version | --list-profiles
```

Входы можно передавать несколькими `-i/--in` или позиционными аргументами.
Один `-o <файл>` допустим только для одного входа; для нескольких входов
используйте `-o <папка>`, `--out-dir` или `--out-ext`.

---

## Входы и выходы

### Входы

* `-i, --in <path>` — файл или папка; можно повторять.
* Позиционные аргументы без `-` тоже считаются входами (удобно для drag&drop).
* Папка разворачивается в видеофайлы **верхнего уровня** и сортируется по имени.
* Поддерживаемые расширения папки: `.mkv`, `.mp4`, `.m4v`, `.mov`, `.ts`,
  `.m2ts`, `.webm`, `.avi`, `.wmv`, `.mpg`, `.mpeg`, `.flv`.
* Один и тот же файл, указанный дважды, не дублируется (путь нормализуется).

### Выход

`-o/--out <path>`:

* файл с известным расширением — один выходной файл;
* существующая папка или путь с `\`/`/` на конце — папка выхода;
* для нескольких входов `-o` обязан быть папкой; известное файловое
  расширение при нескольких входах — ошибка;
* `-o` и `--out-dir` одновременно нельзя.

`--out-dir <dir>` — папка выхода для пакетного режима. Если не задана,
используется `<папка входа>\upscaled`.

`--out-ext <ext>` — расширение для пакетных имён:

* `mp4` (по умолчанию), `mkv`, `h264`, `h265`, `av1`, `raw`;
* `h265` требует `--codec-out hevc` (или `h265`), `h264` — `h264`,
  `av1` — `av1`;
* `raw` означает «сырой поток по кодеку»: `h264` -> `.h264`,
  `hevc`/`h265` -> `.h265`, `av1` -> `.av1`;
* `hevc` — алиас `h265`;
* `--out-ext` нельзя использовать вместе с `-o <файл>`; с `-o <папка>`
  и с `--out-dir` — можно.

Имя пакетного выхода: `<stem>.<ext>`. Коллизии (включая регистр кириллицы)
и совпадение выхода с любым входом проверяются до запуска.

---

## Флаги

### Основные

| Флаг | Значение | По умолчанию | Примечание |
|---|---|---|---|
| `-i, --in <path>` | файл/папка | — | можно повторять |
| `-e, --engine <path>` | готовый engine | — | engine привязан к GPU/TRT |
| `--onnx <path>` | ONNX-модель | — | сборка/кэш engine |
| `--rebuild-engine` | флаг | выкл | требует `--onnx`; в пакете один раз на ключ |
| `-o, --out <path>` | файл/папка | — | один вход — файл; несколько — папка |
| `--out-dir <dir>` | папка | `<вход>\upscaled` | нельзя с `-o` |
| `--out-ext <ext>` | `mp4, mkv, h264, h265, av1, raw` | `mp4` | только пакет/папка |
| `--codec-out <c>` | `h264, hevc, h265, av1` | `h264` | AV1 — только raw |
| `--audio <copy/none>` | аудио | `copy` | `--no-audio` — алиас `none` |
| `--subs <copy/none>` | субтитры | `copy` | `--no-subs` — алиас `none` |
| `--profile <name>` | `fast, balanced, quality, quality-split` | defaults | явные флаги перекрывают |
| `--scale <1..8>` | ожидаемый множитель | `2` | фактический берётся из engine |
| `--frames <N>` | остановиться после N кадров | `0` (все) | диагностический |
| `--max-frames <N>` | алиас `--frames` | `0` | скрытый алиас |
| `--gray` | флаг | выкл | серый кадр вместо compute; engine всё равно нужен |
| `--live` | флаг | выкл | зарезервирован, поведения нет |
| `--dry-run` | флаг | выкл | показать конфиг и выйти |
| `--help`, `-h`, `-?`, `/?` | флаг | — | справка |
| `--version`, `-V` | флаг | — | версия |
| `--list-profiles` | флаг | — | список профилей |

### Энкодер

| Флаг | Значение | По умолчанию | Примечание |
|---|---|---|---|
| `--preset <p1..p7/quality/performance>` | пресет NVENC | `p1` | `quality` -> P7, `performance` -> P1 |
| `--tune <ull/ll/hq/quality>` | tuning | `ull` | `quality` = `hq` |
| `--rc <cbr/cqp/constqp>` | rate control | `cbr` | `constqp` — алиас CQP |
| `--qp <0..51>` | QP для CQP | `0` | используется только CQP |
| `--bitrate <kbps>` | битрейт CBR | `25000` | 1..1 000 000 |
| `--gop <1..10000>` | GOP | `120` | |
| `--bframes <0..5>` | B-кадры | `0` | >5 обрезается |
| `--ref <0..16>` | ref/DPB | `0` (авто) | >16 обрезается |
| `--lookahead <0..31>` | lookahead | `0` | обрезается до `31-bframes` |
| `--bref-mode <middle/disabled/auto>` | B-ref | disabled | `auto` = disabled |
| `--encoders <1/2>` | NVENC-сессии | `2` | chunked dual только B=0/lookahead=0 |
| `--split-enc <auto/auto_forced/forced_2/forced_3/disable>` | split | `disable` | H.264 не поддерживает |
| `--bit-depth <8/10>` | глубина | `8` | 10 = P010 / HEVC Main10 |
| `--10bit` | флаг | — | алиас `--bit-depth 10` |

Ограничения, которые применяются автоматически:

* H.264 + 10-bit -> 8-bit (с предупреждением).
* H.264 + split -> split отключается.
* B-кадры/lookahead + `--encoders 2` -> одна сессия.
* split + `--encoders 2` -> одна сессия.
* B-кадры + raw-выход -> ошибка CLI (см. ниже).

---

## Профили

| Профиль | Кодек | Preset/tune | RC | B | ref/LA | Глубина | Сессии |
|---|---|---|---|---|---|---|---|
| `fast` | HEVC | P1/HQ | CQP18 | 0 | 0/0 | 8-bit | 2, chunked |
| `balanced` | HEVC | P7/HQ | CQP18 | 0 | 0/0 | 10-bit | 2, chunked |
| `quality` | HEVC | P7/HQ | CQP18 | 5 | 8/16 | 10-bit | 1 |
| `quality-split` | HEVC | P7/HQ | CQP18 | 5 | 8/16 | 10-bit | 1 + split forced_2 |

Все профили ставят GOP 250; у `quality*` включён `bref-middle`.

Явные флаги перекрывают профиль **независимо от порядка**:

```powershell
nisshoku --profile quality --bframes 3 --qp 20 ...
nisshoku --bframes 2 --profile quality ...
```

В обоих случаях применяются явные `3/20` или `2`; остальное — из профиля.

---

## Пакетный режим

* Несколько `-i`, позиционные входы или папка — каждый файл обрабатывается
  последовательно.
* Выход по умолчанию: `<папка входа>\upscaled\<stem>.mp4`.
* `--out-dir` задаёт общую папку; `--out-ext` — расширение.
* Коллизии выходов и совпадение выхода с любым входом — ошибка до запуска.
* Ошибка одного файла не останавливает остальные.
* Итог: `[batch] итог: ok=N failed=M`; exit 1, если `failed > 0`.
* При ошибке файла его частичный выход может остаться на диске; повторный
  запуск перезапишет его.

Пример:

```powershell
nisshoku -i a.mkv -i b.mp4 --onnx model.onnx --out-dir "D:\out" --out-ext mkv
```

---

## Форматы и матрица выходов

| Выход | Контейнер | Видео | Аудио | Субтитры | B>0 | Примечание |
|---|---|---|---|---|---|---|
| `.mp4` | MP4 | H.264 / HEVC | copy | **не копируются** | да | AV1 запрещён; edit list для B |
| `.mkv` | MKV | H.264 / HEVC | copy | copy | да | AV1 запрещён |
| `.h264` | raw | H.264 | нет | нет | **нет** | только B=0 |
| `.h265`, `.hevc` | raw | HEVC | нет | нет | **нет** | только B=0 |
| `.av1` | raw | AV1 | нет | нет | **нет** | только raw |
| `.raw`, без расширения | raw | по `--codec-out` | нет | нет | **нет** | кодек задаёт `--codec-out` |

Правила:

* Расширение `-o` должно соответствовать кодеку: `.h265/.hevc` ->
  `--codec-out hevc` (или `h265`), `.h264` -> `h264`, `.av1` -> `av1`.
* `.raw`/без расширения — сырой поток; кодек берётся из `--codec-out`.
* AV1 в `.mkv`/`.mp4` запрещён; используйте `.av1` или HEVC/H.264.
* Raw-выход при `bframes > 0` запрещён: raw-демуксер теряет порядок PTS.
* H.264 10-bit автоматически становится 8-bit.
* 10-bit HEVC — P010 / Main10; 8-bit — NV12.
* MP4 не копирует ASS/PGS/прочие субтитры; MKV копирует.

---

## Движок: engine, ONNX и кэш

Порядок выбора:

1. `--rebuild-engine` + `--onnx`: готовый engine и кэш игнорируются.
2. `-e/--engine`: если файл существует — загружается; если не подошёл и есть
   `--onnx` — продолжается выбор/сборка из ONNX.
3. `<stem>.engine` рядом с ONNX.
4. Кэш `<stem>_<W>x<H>_<gpu>_fp16_trt<ver>.engine` рядом с ONNX.
5. Fallback-кэш `%LOCALAPPDATA%\NisshokuSR\engines`.
6. Сборка из ONNX, сохранение кэша, загрузка в память.

Особенности:

* `--rebuild-engine` без `--onnx` — ошибка CLI (код 2).
* В пакете rebuild выполняется один раз на ключ «ONNX + разрешение + GPU».
* Кэш проверяет путь/размер/mtime ONNX, GPU, TensorRT и входное разрешение.
* ONNX: ровно 1 вход/1 выход, NCHW, 3 канала, батч 1 или динамический.
* Статический вход ONNX должен совпадать с видео; динамический получает
  профиль `[1,3,H,W]`.
* I/O engine строго FP16; FP32-IO при сборке из ONNX принудительно
  переводится в FP16.
* Engine привязан к GPU и версии TensorRT. Portable не содержит engine;
  добавьте свой `-e` или `--onnx`.
* Для сборки из ONNX нужен `nvinfer_builder_resource_10.dll` (~2 ГБ).
  В installer/portable он включён по умолчанию; сборка с
  `NISSHOKU_SKIP_BUILDER=1` умеет только готовые engine.

---

## Аудио и субтитры

* `--audio copy` (по умолчанию) — аудио из исходника копируется.
* `--audio none` или `--no-audio` — без аудио.
* `--subs copy` (по умолчанию) — субтитры копируются в MKV.
* `--subs none` или `--no-subs` — без субтитров.
* MP4 субтитры не копирует; это не ошибка.
* Raw-выход аудио/субтитры не содержит.
* При `--frames N` аудио/субтитры обрезаются по последнему кадру.
  Для Opus обрезка сейчас неточная — известный баг, см. README.

---

## Диагностика и ограничения

### `--frames N`

Останавливается после N закодированных кадров; `0` — все. Предназначен для
диагностики и коротких проверок, не для точного trim производства.
`--max-frames` — скрытый алиас.

### `--gray`

Пропускает compute/TRT-инференс и заливает серый кадр, но **engine всё равно
инициализируется**: выходное разрешение берётся из engine. Это проверка
decode -> fill -> encode, а не безмодельный режим.

### `--scale`

Задаёт ожидаемый множитель 1..8. Фактический масштаб берётся из engine;
если не совпадает, CLI печатает предупреждение, но не масштабирует картинку.

### `--live`

Зарезервирован. Устанавливает флаг в конфиг, но отдельного live-режима нет.

---

## Прогресс и коды возврата

CLI печатает прогресс примерно раз в 2 секунды:

```text
[progress] frames=N total=M pct=P fps=F speed=S elapsed=E eta=T
```

В пакете добавляются `file=K files=L`:

```text
[progress] ... file=2 files=5
```

Финальная строка:

```text
done, frames=N, S s, F fps -> <path>
failed, frames=N, S s, F fps -> <path>
```

`done` при `--frames` — успешное диагностическое завершение, а не ошибка.
GUI парсит `[progress]`; не меняйте формат без обновления `tools/gui.py`.

Коды возврата:

| Код | Значение |
|---|---|
| `0` | успех (включая `--frames`/`--dry-run`/help/version) |
| `1` | ошибка выполнения: demux/decode/engine/NVENC/mux; в пакете хотя бы один failed |
| `2` | неверные аргументы/входы/выходы/флаги |

Ошибки печатаются в stderr; прогресс и итоговые строки — в stdout.

---

## Переменные окружения

| Переменная | Где работает | Смысл |
|---|---|---|
| `NISSHOKU_KEEP_TS=1` | runtime CLI | Оставить временный `<out>.tmp.ts` и лог FFmpeg для разбора |
| `FFMPEG_ROOT` | runtime fallback | Если `ffmpeg.exe` не найден рядом с exe, ищется в `%FFMPEG_ROOT%\bin` |
| `NISSHOKU_SKIP_BUILDER=1` | build_release | Не включать builder resource; portable не сможет собирать engine из ONNX |
| `NISSHOKU_THIRD_PARTY` | build_release/GUI | Другой путь к `third_party` при сборке/запуске GUI |
| `NISSHOKU_EXE` | GUI | Явный путь к `nisshoku.exe` для `tools/gui.py` |

`PATH` установщик не меняет. `TRT_ROOT`, `CUDA_PATH`, `NVCODEC_SDK_DIR` нужны
только для сборки из исходников, не для запуска portable.

---

## Примеры

### Готовый engine

```powershell
nisshoku -i in.mkv -e model.engine -o out.mkv --profile quality
nisshoku -i in.mkv -e model.engine -o out.mp4 --profile quality-split
nisshoku -i in.mkv -e model.engine -o out.h265 --profile fast
```

### ONNX

```powershell
# engine соберётся/подхватится под текущую GPU и разрешение
nisshoku -i in.mkv --onnx model.onnx -o out.mkv --profile quality-split

# форсировать пересборку
nisshoku -i in.mkv --onnx model.onnx -o out.mkv --profile quality --rebuild-engine
```

### Пакет

```powershell
# папка -> <папка>\upscaled\<stem>.mp4
nisshoku -i "D:\серии" --onnx model.onnx --profile quality-split

# несколько файлов в общую папку, MKV
nisshoku -i a.mkv -i b.mp4 --onnx model.onnx --out-dir "D:\out" --out-ext mkv

# один файл, папка выхода
nisshoku -i in.mkv -e model.engine -o "D:\out" --out-ext mkv
```

### Аудио/субтитры

```powershell
nisshoku -i in.mkv -e model.engine -o out.mkv --audio none
nisshoku -i in.mkv -e model.engine -o out.mkv --subs none
nisshoku -i in.mkv -e model.engine -o out.mp4 --audio copy
```

### Диагностика

```powershell
nisshoku -i in.mkv -e model.engine -o gray.mkv --profile fast --gray --frames 60
nisshoku --dry-run --profile quality-split
nisshoku --dry-run -i in.mkv --codec-out hevc -o out.mkv
```

### Полезные ошибки

```powershell
# exit 2: расширение не совпадает с кодеком
nisshoku -i in.mkv -e model.engine -o out.h265

# exit 2: raw нельзя при B-кадрах
nisshoku -i in.mkv -e model.engine -o out.h265 --profile quality

# exit 2: AV1 нельзя в MKV/MP4
nisshoku -i in.mkv -e model.engine -o out.mkv --codec-out av1

# exit 2: --rebuild-engine требует --onnx
nisshoku -i in.mkv -e model.engine -o out.mkv --rebuild-engine
```

---

## Известные ограничения

* `--frames` + Opus: аудио обрезается неточно (открытый баг, см. README).
* Full-range и BT.601 источники не поддерживаются: вход/выход считаются
  limited BT.709.
* AV1 в MKV/MP4 не поддерживается; только raw `.av1`.
* Raw-выход только при `bframes = 0`; CLI это проверяет.
* H.264 10-bit недоступен; автоматически 8-bit.
* H.264 split encoding недоступен.
* B-кадры/lookahead требуют одной NVENC-сессии.
* TensorRT engine привязан к GPU и версии TensorRT.
* Модели, веса и готовые engine в installer/portable не входят.
* GUI в installer/portable не входит.
* Битые пакеты декодера пропускаются; в конце печатается
  `[dec] ошибок декодирования: N`.
* Нет time-trim (`--start/--end`); только `--frames`.
* Нет `--skip-existing`/`--overwrite`: повторный запуск перезаписывает выход.
* Нет JSON-вывода, shell completion и man-page; прогресс — текстовый.

Полный журнал багов и ограничений — в [README.md](../README.md).
