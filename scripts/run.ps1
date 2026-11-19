#Requires -Version 5.1
# Run helper: sets DLL PATH (TRT lib+bin, FFmpeg bin, CUDA bin) and forwards args.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$tp = Join-Path (Split-Path $root -Parent) "third_party"
$trt = (Get-Item env:TRT_ROOT -ErrorAction SilentlyContinue).Value
if (!$trt) { $trt = "$tp\TensorRT-10.8" }
$ff = (Get-Item env:FFMPEG_ROOT -ErrorAction SilentlyContinue).Value
if (!$ff) { $ff = "$tp\ffmpeg-shared\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1" }
$cudaBin = Join-Path $env:CUDA_PATH "bin"
$env:PATH = "$trt\lib;$trt\bin;$ff\bin;$cudaBin;$env:PATH"
& (Join-Path $root "build\nisshoku.exe") @args
exit $LASTEXITCODE
