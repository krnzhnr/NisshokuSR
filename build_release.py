#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сборка portable-папки и Inno Setup installer для NisshokuSR.

Запуск: run_build_release.bat (или python build_release.py).
Результат:
  dist\\NisshokuSR\\                         — автономная portable-папка
  dist\\NisshokuSR_v<version>_portable.zip   — её ZIP
  setup_output\\NisshokuSR_v<version>_Setup.exe — installer (если найден ISCC)
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

# UTF-8 в консоли и дочерних процессах.
os.environ["PYTHONIOENCODING"] = "utf-8"
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
VERSION_FILE = BASE / "version.txt"
DIST = BASE / "dist"
PORTABLE_NAME = "NisshokuSR"
PORTABLE = DIST / PORTABLE_NAME
SETUP_OUT = BASE / "setup_output"
BUILD_SCRIPT = BASE / "scripts" / "build_windows.ps1"
ISS_PATH = BASE / "NisshokuSR.iss"

# Где лежат зависимости при сборке. Можно переопределить env-переменными.
TP = Path(os.environ.get("NISSHOKU_THIRD_PARTY", r"F:\Programing\Utils\third_party"))
TRT_DEFAULT = TP / "TensorRT-10.8"
FFMPEG_DEFAULT = TP / "ffmpeg-shared" / "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1"
CUDA_DEFAULT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")

TRT_DLLS = [
    "nvinfer_10.dll",
    "nvinfer_plugin_10.dll",
    "nvinfer_vc_plugin_10.dll",
    "nvinfer_dispatch_10.dll",
    # Нужен exe для разбора ONNX при сборке engine.
    "nvonnxparser_10.dll",
]
# ~1.97 ГБ: нужен только для сборки engine из ONNX (загружается builder'ом динамически).
TRT_BUILDER_RESOURCE = "nvinfer_builder_resource_10.dll"
FFMPEG_DLLS = [
    "avformat-62.dll",
    "avcodec-62.dll",
    "avutil-60.dll",
    "swresample-6.dll",
    # Нужны встроенному ffmpeg.exe для remux в MKV/MP4.
    "avfilter-11.dll",
    "avdevice-62.dll",
    "swscale-9.dll",
]
VC_DLLS = ["MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]
# Драйверные nvcuda/nvcuvid/nvEncodeAPI64 не кладём: они приходят с NVIDIA driver.


def log(msg: str) -> None:
    print(f"[*] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def run(cmd, **kwargs) -> None:
    log(" ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True, **kwargs)


def read_version() -> str:
    if not VERSION_FILE.exists():
        die(f"нет {VERSION_FILE}")
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        die("version.txt пустой")
    return version


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        die(f"не найден файл: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f"copy {src.name} -> {dst.relative_to(PORTABLE)}")


def find_vc_runtime() -> Path:
    roots = [
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*/*/VC/Redist/MSVC/*/x64/Microsoft.VC143.CRT",
                        "*/*/VC/Redist/MSVC/*/x64/Microsoft.VC142.CRT"):
            for candidate in root.glob(pattern):
                if all((candidate / dll).exists() for dll in VC_DLLS):
                    return candidate
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    log("VC++ redist не найден в Visual Studio, беру System32")
    return system32


def find_iscc() -> Optional[Path]:
    candidates = [
        Path(r"F:\Programs\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe"),
    ]
    which = shutil.which("ISCC")
    if which:
        candidates.insert(0, Path(which))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Последний шанс: реестр Inno Setup.
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1")
                loc, _ = winreg.QueryValueEx(key, "InstallLocation")
                p = Path(loc) / "ISCC.exe"
                if p.exists():
                    return p
            except OSError:
                pass
    except Exception:
        pass
    return None


def write_portable_readme(version: str) -> None:
    text = f"""NisshokuSR {version} — portable
=================================

Требуется:
  * NVIDIA driver (CUDA 12.8 / NVENC API 13.x);
    nvcuda.dll, nvcuvid.dll, nvEncodeAPI64.dll должны быть в System32.
  * exe: RTX 20/30/40/50 (sm_75/86/89/120).
  * Готовый engines\\animejanai_fp16.engine в этой сборке может отсутствовать:
    он не входит в исходный репозиторий. Используйте --onnx или положите
    свой engine в engines\\animejanai_fp16.engine. Engine привязан к GPU/TRT:
    на другой карте запускайте с --onnx, он соберётся под неё автоматически.

Быстрый старт:
  nisshoku.exe --help
  nisshoku.exe --list-profiles
  nisshoku.exe -i in.mkv --onnx model.onnx -o out.mkv --profile quality-split
  nisshoku.exe --dry-run --profile quality-split

Апскейл из ONNX:
  nisshoku.exe -i in.mkv --onnx model.onnx -o out.mp4 --profile quality-split
  Готовый engine под текущие GPU и разрешение подхватывается автоматически.
  Если его нет, engine собирается из ONNX при запуске под текущую GPU и
  разрешение и кэшируется рядом с ONNX
  (<model>_<W>x<H>_<GPU>_fp16_trt<версия>.engine); если папка недоступна
  для записи — в %LOCALAPPDATA%\\NisshokuSR\\engines. Первый запуск с новой
  моделью/GPU/разрешением заметно дольше.
  Для сборки нужен nvinfer_builder_resource_10.dll (~1.97 ГБ) рядом с exe.
  Если файл не включён в portable, доступны только готовые engine
  (--engine или ранее собранный кэш), сборка из ONNX не работает.

Пакетный режим:
  Несколько -i или папка на входе: файлы обрабатываются последовательно,
  имя выхода повторяет исходник. По умолчанию — <папка входа>\\upscaled\\<stem>.mp4.
  Пример:
  nisshoku.exe --profile quality-split -i "D:\\серии" --onnx model.onnx -o "D:\\out"

Профили:
  fast           P1/HQ, B=0, dual NVENC, 8-bit
  balanced       P7/HQ, B=0, dual NVENC, 10-bit
  quality        P7/HQ, B=5, ref=8, lookahead 16, Main10, 1 сессия
  quality-split  quality + split encoding forced_2

Выход .mkv/.mp4 — с аудио и субтитрами (субтитры только в MKV);
для упаковки используется встроенный ffmpeg.exe. .h265/.h264 — сырой поток.
Аудио/субтитры можно отключить: --audio none, --subs none.

CUDA Toolkit, TensorRT SDK и Video Codec SDK ставить не нужно:
их runtime-DLL уже лежат рядом с exe. Нужен только драйвер NVIDIA.

Документация: README.md, docs\\CLI.md.
"""
    (PORTABLE / "PORTABLE.txt").write_text(text, encoding="utf-8")


def write_cmd_wrapper() -> None:
    (PORTABLE / "nisshoku.cmd").write_text(
        '@echo off\r\n"%~dp0nisshoku.exe" %*\r\n', encoding="utf-8")


def write_iss(version: str) -> None:
    iss = f"""; Автоматически сгенерировано build_release.py
[Setup]
AppId=krnzhnr.nisshokusr
AppName=NisshokuSR
AppVersion={version}
AppPublisher=NisshokuSR
DefaultDirName={{autopf}}\\NisshokuSR
DefaultGroupName=NisshokuSR
UsePreviousAppDir=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={SETUP_OUT}
OutputBaseFilename=NisshokuSR_v{version}_Setup
Compression=lzma2/ultra64
SolidCompression=yes
LZMADictionarySize=65536
WizardStyle=modern
UninstallDisplayName=NisshokuSR CLI

[Files]
Source: "{PORTABLE}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\NisshokuSR CLI"; Filename: "{{cmd}}"; __CLI_PARAMS__ WorkingDir: "{{app}}"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if (not FileExists(ExpandConstant('{{sys}}\\nvcuvid.dll'))) or
     (not FileExists(ExpandConstant('{{sys}}\\nvEncodeAPI64.dll'))) then
    MsgBox('Не найден NVIDIA-драйвер: nvcuvid.dll / nvEncodeAPI64.dll.' + #13#10 +
           'Установите свежий драйвер NVIDIA, иначе пайплайн не запустится.',
           mbInformation, MB_OK);
end;
"""
    iss = iss.replace("__CLI_PARAMS__", 'Parameters: "/k ""{app}\\nisshoku.cmd""";')
    ISS_PATH.write_text(iss, encoding="utf-8")
    log(f"write {ISS_PATH.name}")


def main() -> int:
    version = read_version()
    log(f"NisshokuSR {version}")

    env = os.environ.copy()
    env.setdefault("NVCODEC_SDK_DIR", str(TP / "VideoCodecSDK"))
    env.setdefault("TRT_ROOT", str(TRT_DEFAULT))
    env.setdefault("FFMPEG_ROOT", str(FFMPEG_DEFAULT))
    env.setdefault("CUDA_PATH", str(CUDA_DEFAULT))

    log("build")
    run(["powershell", "-ExecutionPolicy", "Bypass", "-File", BUILD_SCRIPT], env=env)

    exe = BASE / "build" / "nisshoku.exe"
    if not exe.exists():
        die(f"нет {exe} после сборки")

    if PORTABLE.exists():
        shutil.rmtree(PORTABLE)
    PORTABLE.mkdir(parents=True)
    DIST.mkdir(parents=True, exist_ok=True)

    copy_required(exe, PORTABLE / "nisshoku.exe")

    cuda_bin = Path(env["CUDA_PATH"]) / "bin"
    trt_lib = Path(env["TRT_ROOT"]) / "lib"
    ff_bin = Path(env["FFMPEG_ROOT"]) / "bin"
    vc_dir = find_vc_runtime()

    copy_required(cuda_bin / "cudart64_12.dll", PORTABLE / "cudart64_12.dll")
    for dll in TRT_DLLS:
        copy_required(trt_lib / dll, PORTABLE / dll)
    if os.environ.get("NISSHOKU_SKIP_BUILDER", "").strip() == "1":
        print(f"[!] NISSHOKU_SKIP_BUILDER=1: {TRT_BUILDER_RESOURCE} не включён — "
              "portable сможет запускать только готовые engine (сборка из ONNX недоступна).",
              file=sys.stderr, flush=True)
    else:
        copy_required(trt_lib / TRT_BUILDER_RESOURCE, PORTABLE / TRT_BUILDER_RESOURCE)
    for dll in FFMPEG_DLLS:
        copy_required(ff_bin / dll, PORTABLE / dll)
    # ffmpeg.exe нужен для финальной упаковки .mkv/.mp4 (аудио/субтитры).
    copy_required(ff_bin / "ffmpeg.exe", PORTABLE / "ffmpeg.exe")
    for dll in VC_DLLS:
        copy_required(vc_dir / dll, PORTABLE / dll)

    engine_src = BASE / "testdata" / "animejanai_fp16.engine"
    if engine_src.exists():
        copy_required(engine_src, PORTABLE / "engines" / "animejanai_fp16.engine")
    else:
        log("sample engine не найден: portable будет без готового engines\\animejanai_fp16.engine; "
            "используйте --onnx или положите свой engine в testdata")

    readme = BASE / "README.md"
    if readme.exists():
        copy_required(readme, PORTABLE / "README.md")
    cli_doc = BASE / "docs" / "CLI.md"
    if cli_doc.exists():
        copy_required(cli_doc, PORTABLE / "docs" / "CLI.md")
    config = BASE / "configs" / "pipeline.json"
    if config.exists():
        copy_required(config, PORTABLE / "configs" / "pipeline.json")

    write_portable_readme(version)
    write_cmd_wrapper()

    zip_path = DIST / f"NisshokuSR_v{version}_portable.zip"
    if zip_path.exists():
        zip_path.unlink()
    log(f"zip -> {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        for path in sorted(PORTABLE.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(Path(PORTABLE_NAME) / path.relative_to(PORTABLE)))

    write_iss(version)
    iscc = find_iscc()
    if iscc:
        SETUP_OUT.mkdir(parents=True, exist_ok=True)
        run([iscc, ISS_PATH])
        installer = SETUP_OUT / f"NisshokuSR_v{version}_Setup.exe"
        if not installer.exists():
            die(f"ISCC отработал, но нет {installer}")
        log(f"installer: {installer} ({installer.stat().st_size / 1e6:.1f} MB)")
    else:
        log("ISCC.exe не найден: portable и ZIP собраны, installer пропущен.")
        log("Установи Inno Setup 6 и запусти скрипт ещё раз.")

    log(f"portable: {PORTABLE}")
    log(f"zip:      {zip_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"[!] команда завершилась с кодом {e.returncode}", file=sys.stderr)
        raise SystemExit(1)
